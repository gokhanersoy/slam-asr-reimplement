#!/usr/bin/env python3
"""Standalone script to apply Zemberek Language Model correction/rescoring to ASR JSONL predictions.

Usage:
    python scripts/apply_zemberek_lm.py \
        --input eval_results.jsonl \
        --output eval_results_zemberek_lm.jsonl \
        --train_manifest data/issai_tsc/train.jsonl
"""

import argparse
import json
import os
import sys

# Add parent directory to path to allow importing slam_asr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slam_asr.utils.zemberek_lm import ZemberekLanguageModel
from slam_asr.utils.metrics import compute_wer, compute_cer, normalize_text_turkish


def parse_args():
    parser = argparse.ArgumentParser(description="Apply Zemberek LM to ASR JSONL predictions")
    parser.add_argument("--input", type=str, required=True, help="Input eval_results.jsonl file")
    parser.add_argument("--output", type=str, default="eval_results_zemberek_lm.jsonl", help="Output JSONL file")
    parser.add_argument("--train_manifest", type=str, default=None, help="Path to training manifest for in-domain LM")
    parser.add_argument("--lm_file", type=str, default=None, help="Optional precomputed LM file (.json)")
    parser.add_argument("--zemberek_jar", type=str, default=None, help="Optional path to zemberek-full.jar")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input predictions file not found: {args.input}")

    manifest_path = args.train_manifest if (args.train_manifest and os.path.exists(args.train_manifest)) else None
    print(f"[*] Initializing Zemberek Language Model...")
    if manifest_path:
        print(f"[*] Building in-domain language model from: {manifest_path}")

    lm = ZemberekLanguageModel(
        manifest_path=manifest_path,
        lm_file=args.lm_file,
        zemberek_jar_path=args.zemberek_jar,
    )
    print(f"[*] Zemberek LM ready: {len(lm.vocab):,} vocabulary terms, {len(lm.bigrams):,} bigrams indexed.")

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"[*] Processing {len(records)} test samples...")
    raw_hypotheses = []
    lm_hypotheses = []
    references = []

    corrected_records = []
    for item in records:
        raw_pred = item.get("raw_hypothesis") or item.get("hypothesis", "")
        ref = item.get("reference", "")

        lm_pred = lm.correct_sentence(raw_pred)

        raw_hypotheses.append(raw_pred)
        lm_hypotheses.append(lm_pred)
        references.append(ref)

        corrected_item = dict(item)
        corrected_item["raw_hypothesis"] = raw_pred
        corrected_item["lm_hypothesis"] = lm_pred
        corrected_item["hypothesis"] = lm_pred
        corrected_item["norm_raw_hypothesis"] = normalize_text_turkish(raw_pred)
        corrected_item["norm_lm_hypothesis"] = normalize_text_turkish(lm_pred)
        corrected_item["norm_hypothesis"] = normalize_text_turkish(lm_pred)
        corrected_records.append(corrected_item)

    raw_wer = compute_wer(raw_hypotheses, references, language="tr")
    raw_cer = compute_cer(raw_hypotheses, references, language="tr")
    lm_wer = compute_wer(lm_hypotheses, references, language="tr")
    lm_cer = compute_cer(lm_hypotheses, references, language="tr")
    wer_gain = raw_wer - lm_wer
    cer_gain = raw_cer - lm_cer

    print("\n" + "=" * 65)
    print(f"[*] Evaluation Comparison on: {args.input}")
    print(f"[*] RAW ASR (Before Zemberek LM):")
    print(f"    • Word Error Rate (WER):      {raw_wer:.2f}%")
    print(f"    • Character Error Rate (CER): {raw_cer:.2f}%")
    print(f"[*] WITH ZEMBEREK LANGUAGE MODEL (Görmez 2024 Support):")
    print(f"    • Word Error Rate (WER):      {lm_wer:.2f}%  (Net Kazanç: -{wer_gain:.2f}% mutlak)")
    print(f"    • Character Error Rate (CER): {lm_cer:.2f}%  (Net Kazanç: -{cer_gain:.2f}% mutlak)")
    print("=" * 65)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in corrected_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[*] Corrected predictions written to: {args.output}")


if __name__ == "__main__":
    main()
