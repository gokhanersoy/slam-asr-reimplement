#!/usr/bin/env python3
"""Standalone script to apply Zemberek Language Model / Rescoring to ASR evaluation results.

References:
- Akin & Akin. "Zemberek Turkish NLP Framework"
- Görmez, Y. (2024). "Customized deep learning based Turkish automatic speech recognition
  system supported by language model." PeerJ Computer Science, 10, e1981.

Usage:
    python scripts/apply_zemberek_lm.py \
        --input eval_results.jsonl \
        --output eval_results_zemberek_lm.jsonl \
        --train_manifest data/issai_tsc/train.jsonl \
        --ngram_order 3
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
    parser = argparse.ArgumentParser(description="Apply Zemberek Language Model to ASR JSONL predictions")
    parser.add_argument("--input", type=str, required=True, help="Input eval_results.jsonl file")
    parser.add_argument("--output", type=str, default="eval_results_zemberek_lm.jsonl", help="Output JSONL file")
    parser.add_argument("--train_manifest", type=str, default="data/issai_tsc/train.jsonl", help="Path to training manifest for in-domain LM n-gram learning")
    parser.add_argument("--zemberek_jar", type=str, default="tools/zemberek-full.jar", help="Optional path to zemberek-full.jar")
    parser.add_argument("--lexicon_file", type=str, default=None, help="Optional text/json file with lexicon words")
    parser.add_argument("--ngram_order", type=int, default=3, help="Max n-gram order (default: 3)")
    parser.add_argument("--lm_weight", type=float, default=0.35, help="LM score interpolation weight (default: 0.35)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input predictions file not found: {args.input}")

    print(f"[*] Initializing Zemberek Language Model (N-Gram Order: {args.ngram_order}, Weight: {args.lm_weight})...")
    manifest_path = args.train_manifest if os.path.exists(args.train_manifest) else None
    if manifest_path:
        print(f"[*] Training in-domain N-Gram language model from: {manifest_path}")
    else:
        print(f"[*] Using built-in core Turkish vocabulary and morphology rules (train manifest not found locally).")

    zemberek_lm = ZemberekLanguageModel(
        zemberek_jar_path=args.zemberek_jar if os.path.exists(args.zemberek_jar) else None,
        train_manifest=manifest_path,
        lexicon_file=args.lexicon_file,
        ngram_order=args.ngram_order,
        lm_weight=args.lm_weight,
    )
    print(f"[*] Zemberek LM ready: {len(zemberek_lm.vocab):,} Turkish words indexed.")

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"[*] Processing {len(records)} test samples...")
    raw_hypotheses = []
    lm_hypotheses = []
    references = []
    processed_records = []

    corrections_made = 0

    for item in records:
        ref = item.get("reference", "") or item.get("norm_reference", "")
        hyp = item.get("raw_hypothesis") or item.get("hypothesis", "")

        lm_hyp = zemberek_lm.correct_sentence(hyp)

        raw_norm = normalize_text_turkish(hyp)
        lm_norm = normalize_text_turkish(lm_hyp)
        ref_norm = normalize_text_turkish(ref)

        if raw_norm != lm_norm:
            corrections_made += 1

        raw_hypotheses.append(raw_norm)
        lm_hypotheses.append(lm_norm)
        references.append(ref_norm)

        updated_item = dict(item)
        updated_item["raw_hypothesis"] = hyp
        updated_item["norm_raw_hypothesis"] = raw_norm
        updated_item["lm_hypothesis"] = lm_hyp
        updated_item["norm_lm_hypothesis"] = lm_norm
        updated_item["hypothesis"] = lm_hyp
        updated_item["norm_hypothesis"] = lm_norm
        updated_item["norm_reference"] = ref_norm
        processed_records.append(updated_item)

    raw_wer = compute_wer(raw_hypotheses, references, normalize=False)
    raw_cer = compute_cer(raw_hypotheses, references, normalize=False)
    lm_wer = compute_wer(lm_hypotheses, references, normalize=False)
    lm_cer = compute_cer(lm_hypotheses, references, normalize=False)

    wer_diff = lm_wer - raw_wer
    cer_diff = lm_cer - raw_cer

    print("\n" + "=" * 70)
    print(f"[*] Evaluation Results on: {args.input}")
    print(f"[*] Total Utterances Evaluated: {len(records)}")
    print(f"[*] Total Sentences Modified by Zemberek LM: {corrections_made}")
    print("-" * 70)
    print(f"[*] RAW ASR (End-to-End Multimodal Model - t5):")
    print(f"    • Word Error Rate (WER):      {raw_wer:.2f}%")
    print(f"    • Character Error Rate (CER): {raw_cer:.2f}%")
    print(f"[*] WITH ZEMBEREK LANGUAGE MODEL RESCORING (t6-lm):")
    print(f"    • Word Error Rate (WER):      {lm_wer:.2f}%  (Fark: {wer_diff:+.2f}%)")
    print(f"    • Character Error Rate (CER): {lm_cer:.2f}%  (Fark: {cer_diff:+.2f}%)")
    print("=" * 70)

    # Print sample predictions
    print("\n[*] Sample Predictions (Zemberek LM Impact):")
    samples_to_show = min(5, len(processed_records))
    for sample in processed_records[:samples_to_show]:
        print(f"  Key: {sample.get('key', 'sample')}")
        print(f"  REF:     {sample['norm_reference']}")
        print(f"  RAW HYP: {sample['norm_raw_hypothesis']}")
        print(f"  LM HYP:  {sample['norm_lm_hypothesis']}")
        print("  " + "-" * 50)

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        for item in processed_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n[*] Corrected predictions successfully written to: {args.output}")


if __name__ == "__main__":
    main()
