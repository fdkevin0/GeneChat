#!/usr/bin/env python3
"""
Generate predictions from a trained GeneChatUnsloth model on a test set.

Usage:
  cd /home/fdkevin/Workspaces/msc_project/GeneChat
  source .venv/bin/activate
  python scripts/eval_generate.py --checkpoint outputs/unsloth_checkpoints/<job_id>/checkpoint_1000.pth --out preds.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Pre-unsloth patches (consolidated — see xpu/patches.py)
# ═══════════════════════════════════════════════════════════════════════
import torch
from xpu.patches import (
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
# Mirror genechat.common.device.attn_implementation() without importing the
# genechat package here — that would pull in unsloth before Phase 1 ordering.
_dev = os.environ.get("GENECHAT_DEVICE", "").lower() or (
    "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available())
    else "cuda" if torch.cuda.is_available() else "cpu")
_gene_config._attn_implementation = "sdpa" if _dev == "cuda" else "eager"

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


def generate(model, gene_seq: str, prompt: str, max_new_tokens: int = 256):
    """Generate a gene function description.

    Reuses the model's own encode_gene/prompt_list_wrap (same code path as
    training) instead of re-deriving the embedding assembly here.

    Returns ``(text, n_new_tokens)`` — the token count is the true number of
    decoded steps (honours early EOS) so callers can report decode throughput.
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
    return text.strip(), int(generated_ids.shape[0])


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
    ap.add_argument("--max-seq-len", type=int, default=3000,
                    help="Truncate input DNA to this many bp (match training ~3000)")
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
    total_tokens = 0          # summed generated tokens (for aggregate tok/s)
    total_gen_time = 0.0      # summed generate() wall-time (excludes I/O/logging)

    for i, gene_id in enumerate(genes):
        if gene_id not in sequences:
            print(f"  [{i+1}/{len(genes)}] {gene_id}: MISSING SEQUENCE — skipping")
            continue

        seq = sequences[gene_id]
        # seq is a list of strings; take the first one
        seq_str = seq[0] if isinstance(seq, list) else seq
        # Match the training distribution: train sequences were capped at
        # ~3000 bp (~6 encode windows). Test sequences reach 1.38 Mbp; feeding
        # 160 kb yields hundreds of gene-embedding tokens the model never saw
        # in training (and encoding them is the dominant cost). Cap to match.
        if len(seq_str) > args.max_seq_len:
            seq_str = seq_str[:args.max_seq_len]

        t_gene = time.time()
        try:
            pred, n_tok = generate(model, seq_str, prompt, max_new_tokens=args.max_new_tokens)
            predictions[gene_id] = pred
        except Exception as e:
            print(f"  [{i+1}/{len(genes)}] {gene_id}: ERROR — {e}", flush=True)
            predictions[gene_id] = ""
            continue

        # Per-gene progress + decode throughput. Decode is slow on XPU (no fused
        # 4-bit kernel — every step re-dequantizes weights), so log tok/s per
        # gene for performance analysis, plus a running ETA, rather than leaving
        # the run silent for minutes at a stretch.
        dt = time.time() - t_gene
        total_tokens += n_tok
        total_gen_time += dt
        tok_s = n_tok / dt if dt > 0 else 0.0
        elapsed = time.time() - start
        eta = (len(genes) - i - 1) * (elapsed / (i + 1))
        print(f"  [{i+1}/{len(genes)}] {gene_id}: {dt:.0f}s  {n_tok}tok  "
              f"{tok_s:.2f}tok/s (ETA {eta/60:.0f}m) | {pred[:60]!r}", flush=True)

    elapsed = time.time() - start
    agg_tok_s = total_tokens / total_gen_time if total_gen_time > 0 else 0.0
    print(f"Done: {len(predictions)} predictions in {elapsed:.0f}s "
          f"({len(predictions)/elapsed:.2f} genes/s)")
    print(f"Decode throughput: {total_tokens} tokens in {total_gen_time:.0f}s "
          f"→ {agg_tok_s:.2f} tok/s avg ({1000/agg_tok_s:.0f} ms/tok)"
          if agg_tok_s > 0 else "Decode throughput: n/a")

    json.dump(predictions, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
