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
# PHASE 1: Pre-unsloth patches (consolidated — see gcu_xpu.py)
# ═══════════════════════════════════════════════════════════════════════
import torch
from gcu_xpu import (
    apply_phase1_patches,
    apply_phase2_patches,
    patch_dnabert2_alibi,
    patch_dnabert2_flash_attn,
)
apply_phase1_patches()

# ═══════════════════════════════════════════════════════════════════════
# Pre-load DNABERT-2 BEFORE unsloth (avoids meta-device crash)
# ═══════════════════════════════════════════════════════════════════════
from transformers import AutoConfig, AutoModel, AutoTokenizer

_gene_config = AutoConfig.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
if not hasattr(_gene_config, "pad_token_id"):
    _gene_config.pad_token_id = 0
_gene_config._attn_implementation = "eager"

try:
    _gene_encoder = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M", config=_gene_config, trust_remote_code=True)
except RuntimeError:
    pass  # Expected crash on first download — code is now cached

patch_dnabert2_alibi()
patch_dnabert2_flash_attn()

_gene_encoder = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M", config=_gene_config, trust_remote_code=True)
_gene_tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)

import genechat.models.genechat_unsloth as _gcu
_gcu._PRELOADED_GENE_ENCODER = _gene_encoder
_gcu._PRELOADED_GENE_TOKENIZER = _gene_tokenizer

# Import model (triggers unsloth import → genechat_unsloth applies its own patches)
from genechat.models.genechat_unsloth import GeneChatUnsloth
from transformers import LlamaTokenizer

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Post-unsloth patches (consolidated)
# ═══════════════════════════════════════════════════════════════════════
apply_phase2_patches()


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


def generate(model, gene_seq: str, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a gene function description.

    Reuses the model's own encode_gene/prompt_list_wrap (same code path as
    training) instead of re-deriving the embedding assembly here.
    """
    device = model._target_device

    gene_embeds, gene_atts = model.encode_gene([gene_seq])
    wrapped_embeds, wrapped_atts = model.prompt_list_wrap(gene_embeds, gene_atts, [prompt])

    bos = torch.ones([1, 1], dtype=torch.long, device=device) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.llama_model.get_input_embeddings()(bos)

    inputs_embeds = torch.cat([bos_embeds, wrapped_embeds], dim=1)
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
