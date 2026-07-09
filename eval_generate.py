#!/usr/bin/env python3
"""
Generate predictions from a trained GeneChatUnsloth model on a test set.

Usage:
  cd /home/fdkevin/Workspaces/msc_project/GeneChat
  source .venv/bin/activate
  python eval_generate.py --checkpoint genechat/unsloth_checkpoints/<job_id>/checkpoint_1000.pth --out preds.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ═══════════════════════════════════════════════════════════════════════
# PHASE 0-1: Same patches as train_unsloth.py
# ═══════════════════════════════════════════════════════════════════════
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
try:
    torch._dynamo.disable()
except Exception:
    pass

if torch.xpu.is_available():
    import torch.utils._triton as _triton
    _triton.is_device_compatible_with_triton = lambda: False
    if not hasattr(torch._C, "_cuda_getCurrentRawStream"):
        torch._C._cuda_getCurrentRawStream = lambda index=None: None
    # Bug #17: mem_get_info mock
    import torch.xpu.memory as _xpu_mem
    _mem_get_info_mock = lambda device=0: (4 * 1024**3, 16 * 1024**3)
    _xpu_mem.mem_get_info = _mem_get_info_mock
    torch.xpu.mem_get_info = _mem_get_info_mock

import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None

# Pre-load DNABERT-2
from transformers import AutoConfig, AutoModel, AutoTokenizer
import glob as _glob

_gene_config = AutoConfig.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
if not hasattr(_gene_config, "pad_token_id"):
    _gene_config.pad_token_id = 0
_gene_config._attn_implementation = "eager"

_dnabert2_cache = os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/"
    "zhihan1996/DNABERT_hyphen_2_hyphen_117M"
)
try:
    _gene_encoder = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M", config=_gene_config, trust_remote_code=True)
except RuntimeError:
    pass

# Patch bert_layers.py
_dnabert2_files = _glob.glob(f"{_dnabert2_cache}/*/bert_layers.py")
if _dnabert2_files:
    with open(_dnabert2_files[0]) as _f:
        _content = _f.read()
    if "if device is None:\n            device = torch.device" not in _content:
        _old = (
            "):\n        # Alibi\n"
            "        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5"
        )
        _new = (
            "):\n        # Alibi\n        if device is None:\n            device = torch.device(\"cpu\")\n"
            "        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5"
        )
        if _old in _content:
            _content = _content.replace(_old, _new)
            with open(_dnabert2_files[0], "w") as _f:
                _f.write(_content)

_gene_encoder = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M", config=_gene_config, trust_remote_code=True)
_gene_tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)

import genechat.models.genechat_unsloth as _gcu
_gcu._PRELOADED_GENE_ENCODER = _gene_encoder
_gcu._PRELOADED_GENE_TOKENIZER = _gene_tokenizer

# Import model (triggers Phase 2 patches in genechat_unsloth)
from genechat.models.genechat_unsloth import GeneChatUnsloth
from transformers import LlamaTokenizer


def load_model(checkpoint_path: str) -> GeneChatUnsloth:
    """Load model from config + checkpoint."""
    model = GeneChatUnsloth(
        freeze_gene_encoder=True,
        gene_model="",
        max_gene_length=160000,
        freeze_adaptor=False,
        freeze_llama=False,
        llama_model="unsloth/Llama-3.1-8B-bnb-4bit",
        max_txt_len=405,
        end_sym="###",
        load_in_4bit=True,
        max_seq_length=2048,
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"  Resumed from iters={ckpt.get('iters', '?')}")

    model.eval()
    return model


def encode_gene(model, seq: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single DNA sequence into gene embeddings."""
    device = model._target_device
    gene_embeds = []
    for i in range(0, len(seq), 512):
        chunk = seq[max(0, min(i, i - 10)):i + 512]
        input_token = model.gene_tokenizer(chunk, return_tensors="pt")["input_ids"]
        input_token = input_token.to(device)
        hidden_states = model.gene_encoder(input_token)[0]
        embedding_mean = torch.mean(hidden_states, dim=1)
        gene_embeds.append(embedding_mean)

    gene_embeds = torch.stack(gene_embeds, axis=1)

    if gene_embeds.dtype != model.hyena_llama_proj.weight.dtype:
        gene_embeds = gene_embeds.to(model.hyena_llama_proj.weight.dtype)

    inputs_llama = model.hyena_llama_proj(
        gene_embeds.squeeze(dim=2)
    ).to(device=device, dtype=torch.bfloat16)

    atts_llama = torch.ones(
        inputs_llama.size()[:-1], dtype=torch.long, device=device
    )
    return inputs_llama, atts_llama


def generate(model, gene_seq: str, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a gene function description."""
    device = model._target_device

    # Encode gene
    gene_embeds, gene_atts = encode_gene(model, gene_seq)

    # Split prompt at <geneHere>
    if "<geneHere>" in prompt:
        p_before, p_after = prompt.split("<geneHere>", 1)
    else:
        p_before, p_after = prompt, ""

    # Tokenize prompt parts
    p_before_tokens = model.llama_tokenizer(
        p_before, return_tensors="pt", add_special_tokens=False,
    ).to(device)
    p_before_embeds = model.llama_model.get_input_embeddings()(
        p_before_tokens.input_ids
    )

    p_after_tokens = model.llama_tokenizer(
        p_after, return_tensors="pt", add_special_tokens=True, padding=True,
    ).to(device)
    p_after_embeds = model.llama_model.get_input_embeddings()(
        p_after_tokens.input_ids
    )

    # BOS embedding
    bos = torch.ones([1, 1], dtype=torch.long, device=device) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.llama_model.get_input_embeddings()(bos)

    # Concatenate: bos + prompt_before + gene + prompt_after
    inputs_embeds = torch.cat(
        [bos_embeds, p_before_embeds, gene_embeds, p_after_embeds], dim=1
    )
    inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16)

    with torch.no_grad():
        outputs = model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=model.llama_tokenizer.pad_token_id,
            eos_token_id=model.llama_tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip input tokens)
    input_len = inputs_embeds.shape[1]
    generated_ids = outputs[0][input_len:]
    text = model.llama_tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description="Generate GeneChat predictions for eval")
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--out", required=True, help="Output predictions JSON")
    ap.add_argument("--test-dir", default="../data_GeneChat/data/test",
                    help="Directory with test seq.json + qa_summary_rule.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of test genes (for quick testing)")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="Max tokens to generate per sample")
    args = ap.parse_args()

    # Load test data
    test_dir = os.path.abspath(args.test_dir)
    seq_path = os.path.join(test_dir, "seq.json")
    qa_path = os.path.join(test_dir, "qa_summary_rule.json")

    sequences = json.load(open(seq_path))
    ground_truth = json.load(open(qa_path))

    # Build gene_id → sequence mapping
    # Build gene_id → summary mapping
    gt_map = {str(r["Gene Id"]): r["Summary"] for r in ground_truth}

    # Load model
    model = load_model(args.checkpoint)

    prompt = "###Human: <gene><geneHere></gene> Please provide a detailed description of the gene. ###Assistant:"

    predictions = {}
    genes = list(gt_map.keys())
    if args.limit:
        genes = genes[:args.limit]

    print(f"Generating predictions for {len(genes)} genes...")
    start = time.time()

    for i, gene_id in enumerate(genes):
        if gene_id not in sequences:
            print(f"  [{i+1}/{len(genes)}] {gene_id}: MISSING SEQUENCE — skipping")
            continue

        seq = sequences[gene_id]
        # seq is a list of strings; take the first one
        seq_str = seq[0] if isinstance(seq, list) else seq
        if len(seq_str) > 160000:
            seq_str = seq_str[:159999]

        try:
            pred = generate(model, seq_str, prompt, max_new_tokens=args.max_new_tokens)
            predictions[gene_id] = pred
        except Exception as e:
            print(f"  [{i+1}/{len(genes)}] {gene_id}: ERROR — {e}")
            predictions[gene_id] = ""

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(genes) - i - 1) / rate
            print(f"  [{i+1}/{len(genes)}] {gene_id}: {pred[:80]}... "
                  f"({rate:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"Done: {len(predictions)} predictions in {elapsed:.0f}s "
          f"({len(predictions)/elapsed:.2f}/s)")

    json.dump(predictions, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
