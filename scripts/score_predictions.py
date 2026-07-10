#!/usr/bin/env python3
"""
Score GeneChat predictions against reference summaries.

Reuses the metric definitions from eval.py (BLEU 1-4 + SimCSE cosine), but
reads the fork's artifacts instead of the original author's hardcoded
/data2/... paths:
  - predictions: {gene_id: prediction_text}   (from eval_generate.py)
  - references:  test qa_summary_rule.json     ([{Gene Id, Summary}, ...])

Usage:
  uv run --no-sync python score_predictions.py \
      --preds preds.json \
      --test-dir ../data_GeneChat/data/test \
      --out scores.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

SIMCSE_PATH = "princeton-nlp/sup-simcse-roberta-large"
BLEU_WEIGHTS = [
    (1.0, 0, 0, 0),
    (1.0 / 2, 1.0 / 2, 0, 0),
    (1.0 / 3, 1.0 / 3, 1.0 / 3, 0),
    (1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4),
]


def embed(model, tokenizer, texts, batch_size=16):
    """Mean-pooler SimCSE embeddings for a list of texts."""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        tokens = tokenizer(
            chunk, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            emb = model(**tokens, output_hidden_states=True, return_dict=True).pooler_output
        out.append(emb.cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser(description="Score GeneChat predictions")
    ap.add_argument("--preds", required=True, help="predictions JSON (gene_id -> text)")
    ap.add_argument("--test-dir", default="../data_GeneChat/data/test")
    ap.add_argument("--out", default="scores.json")
    args = ap.parse_args()

    preds = json.load(open(args.preds))
    qa_path = os.path.join(os.path.abspath(args.test_dir), "qa_summary_rule.json")
    references = json.load(open(qa_path))
    ref_map = {str(r["Gene Id"]): r["Summary"] for r in references}

    # Pair predictions with their reference; skip empty predictions.
    pairs = []
    for gene_id, pred in preds.items():
        ref = ref_map.get(str(gene_id))
        if ref is None or not pred.strip():
            continue
        pairs.append({"gene_id": gene_id, "correct_func": ref, "predict_func": pred})

    print(f"Scoring {len(pairs)} prediction/reference pairs "
          f"({len(preds)} predictions, {len(pairs)} usable)")

    print("Loading SimCSE model...")
    tokenizer = AutoTokenizer.from_pretrained(SIMCSE_PATH)
    model = AutoModel.from_pretrained(SIMCSE_PATH)
    model.eval()

    refs = [p["correct_func"] for p in pairs]
    cands = [p["predict_func"] for p in pairs]
    ref_emb = embed(model, tokenizer, refs)
    cand_emb = embed(model, tokenizer, cands)

    smooth = SmoothingFunction().method1
    bleu_lists = [[] for _ in range(4)]
    simcse_list = []
    for i, p in enumerate(pairs):
        correct = p["correct_func"].split()
        predict = p["predict_func"].split()
        bleu = sentence_bleu(
            [correct], predict, weights=BLEU_WEIGHTS, smoothing_function=smooth
        )
        sim = 1 - cosine(ref_emb[i], cand_emb[i])
        p["bleu"] = list(bleu)
        p["simcse"] = float(sim)
        for n in range(4):
            bleu_lists[n].append(bleu[n])
        simcse_list.append(sim)

    scores = {
        f"average_bleu_{n + 1}": float(np.mean(bleu_lists[n])) for n in range(4)
    }
    scores["average_simcse"] = float(np.mean(simcse_list))
    scores["num_samples"] = len(pairs)

    print("\n===== Average scores =====")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    json.dump({"scores": scores, "per_sample": pairs}, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
