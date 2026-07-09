"""
GeneChat model adapted for Unsloth + Llama 3.1 8B on Intel Arc A770 (XPU).

Key changes vs. the original genechat.py:
  - LlamaForCausalLM → FastLanguageModel (Unsloth's 4-bit QLoRA loader)
  - LoRA via FastLanguageModel.get_peft_model() instead of peft.get_peft_model()
  - load_in_4bit=True on XPU (bf16 would OOM on 16GB A770)
  - All 7 linear layers targeted (q/k/v/o/gate/up/down) per Unsloth best-practice
  - Device-agnostic: uses self.device instead of torch.cuda.current_device()

Architecture stays the same:
  DNABERT-2 (117M, frozen) → Linear Projection → Llama 3.1 8B (4-bit QLoRA)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List

import torch
import torch.nn as nn

import gcu_device as genechat_device

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Pre-unsloth patches (consolidated)
# ═══════════════════════════════════════════════════════════════════════
from gcu_xpu import apply_phase1_patches
apply_phase1_patches()

# Now safe: unsloth detects XPU natively via torch.xpu.is_available()
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Post-unsloth patches (consolidated)
# ═══════════════════════════════════════════════════════════════════════
from gcu_xpu import apply_phase2_patches
apply_phase2_patches()

from genechat.common.registry import registry
from genechat.models.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, AutoModel
from transformers import LlamaTokenizer


@registry.register_model("genechat_unsloth")
class GeneChatUnsloth(Blip2Base):
    """
    BLIP2-style multimodal model: DNA sequence encoder + LLM.
    Same architecture as GeneChat but uses Unsloth for memory-efficient
    4-bit QLoRA fine-tuning of the LLM component.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_llama31_8b": "",
    }

    def __init__(
        self,
        freeze_gene_encoder: bool = True,
        gene_model: str = "",
        max_gene_length: int = 160000,
        freeze_adaptor: bool = False,
        freeze_llama: bool = True,
        llama_model: str = "unsloth/Llama-3.1-8B",
        max_txt_len: int = 405,
        end_sym: str = "###",
        load_in_4bit: bool = True,
        max_seq_length: int = 2048,
        # LoRA hyperparams (used when freeze_llama=False)
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: List[str] | None = None,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self._target_device = genechat_device.device()

        # ── Gene Encoder (DNABERT-2, frozen) ──────────────────────────
        logging.info("Loading Gene Encoder — DNABERT-2 ...")
        self.max_gene_length = max_gene_length

        # Support pre-loaded gene encoder (loaded before unsloth to avoid
        # meta-device issue with torch.compile in DNABERT-2's ALiBi code)
        _mod = sys.modules[__name__]
        if hasattr(_mod, "_PRELOADED_GENE_ENCODER"):
            self.gene_encoder = _mod._PRELOADED_GENE_ENCODER
            self.gene_tokenizer = _mod._PRELOADED_GENE_TOKENIZER
            logging.info("Using pre-loaded DNABERT-2 (before unsloth)")
        else:
            from transformers import AutoConfig
            _gene_config = AutoConfig.from_pretrained(
                "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            if not hasattr(_gene_config, "pad_token_id"):
                _gene_config.pad_token_id = 0
            _gene_config._attn_implementation = "eager"
            self.gene_encoder = AutoModel.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                config=_gene_config, trust_remote_code=True,
                low_cpu_mem_usage=False, _fast_init=False,
            )
            self.gene_tokenizer = AutoTokenizer.from_pretrained(
                "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.gene_encoder = self.gene_encoder.to(self._target_device)

        if freeze_gene_encoder:
            for name, param in self.gene_encoder.named_parameters():
                param.requires_grad = False
            self.gene_encoder = self.gene_encoder.eval()
            self.gene_encoder.train = disabled_train
            logging.info("Gene encoder frozen")
        else:
            self.gene_encoder = self.gene_encoder.train()

        # ── LLM (Llama 3.1 8B via Unsloth FastLanguageModel) ────────
        logging.info("Loading LLM — Llama 3.1 8B (Unsloth 4-bit QLoRA) ...")
        self.llama_model, _ = FastLanguageModel.from_pretrained(
            model_name=llama_model,
            max_seq_length=max_seq_length,
            dtype=None,                     # auto-detect (bf16 on XPU)
            load_in_4bit=load_in_4bit,
            token=os.environ.get("HF_TOKEN"),
            device_map={"": torch.xpu.current_device()},
        )
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            llama_model, use_fast=False
        )
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        # ── Apply LoRA or freeze ─────────────────────────────────────
        if freeze_llama:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            logging.info("LLM frozen (no LoRA)")
        else:
            if lora_target_modules is None:
                lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]
            self.llama_model = FastLanguageModel.get_peft_model(
                self.llama_model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            logging.info(
                "LoRA applied: r=%d, alpha=%d, targets=%s",
                lora_r, lora_alpha, lora_target_modules,
            )

        # ── Projection layer (gene embed → LLM hidden space) ─────────
        gene_embed_dim = (
            self.gene_encoder.embeddings.word_embeddings.weight.shape[1]
        )
        llama_hidden = self.llama_model.config.hidden_size
        self.hyena_llama_proj = nn.Linear(
            gene_embed_dim, llama_hidden, dtype=torch.bfloat16,
        )
        self.hyena_llama_proj = self.hyena_llama_proj.to(self._target_device)

        if freeze_adaptor:
            for name, param in self.hyena_llama_proj.named_parameters():
                param.requires_grad = False
            logging.info("Adaptor frozen")

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    # ── Gene Encoding (DNABERT-2) ────────────────────────────────────
    def encode_gene(self, seqs):
        batch_seqs = [seq for seq in seqs]
        gene_embeds = []

        for seq in batch_seqs:
            for i in range(0, len(seq), 512):
                input_token = self.gene_tokenizer(
                    seq[max(0, min(i, i - 10)):i + 512],
                    return_tensors="pt",
                )["input_ids"]
                input_token = input_token.to(self._target_device)

                hidden_states = self.gene_encoder(input_token)[0]
                embedding_mean = torch.mean(hidden_states, dim=1)
                gene_embeds.append(embedding_mean)

        gene_embeds = torch.stack(gene_embeds, axis=1)

        if gene_embeds.dtype != self.hyena_llama_proj.weight.dtype:
            gene_embeds = gene_embeds.to(self.hyena_llama_proj.weight.dtype)

        inputs_llama = self.hyena_llama_proj(
            gene_embeds.squeeze(dim=2)
        ).to(device=gene_embeds.device, dtype=torch.bfloat16)

        atts_llama = torch.ones(
            inputs_llama.size()[:-1], dtype=torch.long, device=gene_embeds.device
        )
        return inputs_llama, atts_llama

    # ── Prompt wrapping ──────────────────────────────────────────────
    def prompt_list_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            p_before_lst, p_after_lst = [], []
            for p in prompt:
                p_before, p_after = p.split("<geneHere>")
                p_before_lst.append(p_before)
                p_after_lst.append(p_after)

            p_before_tokens = self.llama_tokenizer(
                p_before_lst, return_tensors="pt", add_special_tokens=False,
            ).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", add_special_tokens=True,
                padding=True,
            ).to(img_embeds.device)

            p_before_embeds = self.llama_model.get_input_embeddings()(
                p_before_tokens.input_ids
            )
            p_after_embeds = self.llama_model.get_input_embeddings()(
                p_after_tokens.input_ids
            )

            wrapped_img_embeds = torch.cat(
                [p_before_embeds, img_embeds, p_after_embeds], dim=1
            )
            wrapped_atts_img = atts_img[:, :1].expand(
                -1, wrapped_img_embeds.shape[1]
            )
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    # ── Forward pass ─────────────────────────────────────────────────
    def forward(self, samples):
        seqs = samples["seq"][0]
        gene_embeds, atts = self.encode_gene(seqs)

        img_embeds, atts_img = self.prompt_list_wrap(
            gene_embeds, atts, samples["prompt"]
        )

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(gene_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones(
                [atts_img.shape[0], atts_img.shape[1] + 1],
                dtype=torch.long, device=gene_embeds.device,
            ).fill_(-100)  # +1 for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.get_input_embeddings()(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.get_input_embeddings()(
            to_regress_tokens.input_ids
        )
        inputs_embeds = torch.cat(
            [bos_embeds, img_embeds, to_regress_embeds], dim=1
        )
        attention_mask = torch.cat(
            [atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1
        )

        # Unsloth Llama 3.1 8B weights are bf16 — autocast must match
        with self.maybe_autocast(dtype=torch.bfloat16):
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds.to(dtype=torch.bfloat16),
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss
        return {"loss": loss}

    # ── Factory from config ──────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg):
        llama_model = cfg.get("llama_model", "unsloth/Llama-3.1-8B")
        gene_model = cfg.get("gene_model", "hyenadna-medium-160k-seqlen")
        freeze_gene_encoder = cfg.get("freeze_gene_encoder", True)
        max_gene_length = cfg.get("max_gene_length", 160000)
        freeze_adaptor = cfg.get("freeze_adaptor", True)
        freeze_llama = cfg.get("freeze_llama", True)
        max_txt_len = cfg.get("max_txt_len", 405)
        end_sym = cfg.get("end_sym", "###")
        load_in_4bit = cfg.get("load_in_4bit", True)
        max_seq_length = cfg.get("max_seq_length", 2048)
        lora_r = cfg.get("lora_r", 16)
        lora_alpha = cfg.get("lora_alpha", 16)
        lora_dropout = cfg.get("lora_dropout", 0.0)
        lora_target_modules = cfg.get("lora_target_modules", None)

        model = cls(
            freeze_gene_encoder=freeze_gene_encoder,
            gene_model=gene_model,
            max_gene_length=max_gene_length,
            freeze_adaptor=freeze_adaptor,
            freeze_llama=freeze_llama,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )

        stage1_ckpt = cfg.get("stage1_ckpt", "")
        if stage1_ckpt:
            logging.info("Load gene-encoder + adaptor checkpoint: %s", stage1_ckpt)
            ckpt = torch.load(stage1_ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)

        peft_ckpt = cfg.get("peft_ckpt", "")
        if peft_ckpt:
            logging.info("Load LoRA checkpoint: %s", peft_ckpt)
            ckpt = torch.load(peft_ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)

        return model
