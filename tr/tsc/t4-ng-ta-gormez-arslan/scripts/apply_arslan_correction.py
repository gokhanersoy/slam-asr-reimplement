#!/usr/bin/env python3
"""Standalone script to apply Recep Sinan Arslan Output Correction to ASR evaluation results.

Usage:
    python scripts/apply_arslan_correction.py \
        --input eval_results.jsonl \
        --output eval_results_arslan_corrected.jsonl \
        --train_manifest data/issai_tsc/train.jsonl \
        --threshold 0.33
"""

import argparse
import json
import os
import sys

# Add parent directory to path to allow importing slam_asr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slam_asr.utils.arslan_corrector import ArslanOutputCorrector
from slam_asr.utils.metrics import compute_wer, compute_cer, normalize_text_turkish


def parse_args():
    parser = argparse.ArgumentParser(description="Apply Arslan Output Correction to ASR JSONL predictions")
    parser.add_argument("--input", type=str, required=True, help="Input eval_results.jsonl file")
    parser.add_argument("--output", type=str, default="eval_results_arslan_corrected.jsonl", help="Output JSONL file")
    parser.add_argument("--train_manifest", type=str, default="data/issai_tsc/train.jsonl", help="Path to training manifest for in-domain lexicon")
    parser.add_argument("--lexicon_file", type=str, default=None, help="Optional text/json file with lexicon words")
    parser.add_argument("--threshold", type=float, default=0.33, help="Max Normalized Levenshtein distance (default: 0.33)")
    parser.add_argument("--min_word_len", type=int, default=4, help="Min word length to correct (default: 4)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input predictions file not found: {args.input}")

    print(f"[*] Initializing Arslan Output Corrector (Threshold: {args.threshold}, Min Word Len: {args.min_word_len})...")
    manifest_path = args.train_manifest if os.path.exists(args.train_manifest) else None
    if manifest_path:
        print(f"[*] Building in-domain lexicon from manifest: {manifest_path}")
    else:
        print(f"[*] Using built-in core Turkish lexicon (train manifest not found locally).")

    corrector = ArslanOutputCorrector(
        lexicon_file=args.lexicon_file,
        manifest_path=manifest_path,
        max_nl_distance=args.threshold,
        min_word_len=args.min_word_len,
        apply_colloquial_rules=True,
        apply_spelling_variants=True,
        apply_conjunction_rules=True,
    )
    print(f"[*] Lexicon loaded: {len(corrector.vocab):,} unique Turkish words indexed.")

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"[*] Processing {len(records)} test samples...")
    raw_hypotheses = []
    corrected_hypotheses = []
    references = []
    corrected_records = []

    corrections_made = 0

    for item in records:
        ref = item.get("reference", "") or item.get("norm_reference", "")
        # Accept 'hypothesis' or 'raw_hypothesis'
        hyp = item.get("raw_hypothesis") or item.get("hypothesis", "")

        corrected_hyp = corrector.correct_sentence(hyp)

        raw_norm = normalize_text_turkish(hyp)
        corr_norm = normalize_text_turkish(corrected_hyp)
        ref_norm = normalize_text_turkish(ref)

        if raw_norm != corr_norm:
            corrections_made += 1

        raw_hypotheses.append(raw_norm)
        corrected_hypotheses.append(corr_norm)
        references.append(ref_norm)

        updated_item = dict(item)
        updated_item["raw_hypothesis"] = hyp
        updated_item["norm_raw_hypothesis"] = raw_norm
        updated_item["corrected_hypothesis"] = corrected_hyp
        updated_item["norm_corrected_hypothesis"] = corr_norm
        updated_item["norm_reference"] = ref_norm
        corrected_records.append(updated_item)

    raw_wer = compute_wer(raw_hypotheses, references, normalize=False)
    raw_cer = compute_cer(raw_hypotheses, references, normalize=False)

    corr_wer = compute_wer(corrected_hypotheses, references, normalize=False)
    corr_cer = compute_cer(corrected_hypotheses, references, normalize=False)

    wer_diff = raw_wer - corr_wer
    cer_diff = raw_cer - corr_cer
    rel_wer_reduction = (wer_diff / raw_wer) * 100.0 if raw_wer > 0 else 0.0

    print("\n" + "=" * 65)
    print("      RECEP SİNAN ARSLAN OUTPUT CORRECTION BENCHMARK REPORT      ")
    print("=" * 65)
    print(f"Total Sentences Evaluated:   {len(records):,}")
    print(f"Sentences Modified:          {corrections_made:,} ({100.0 * corrections_made / len(records):.2f}%)")
    print("-" * 65)
    print(f"RAW SLAM-ASR:")
    print(f"  • Word Error Rate (WER):     {raw_wer:.2f}%")
    print(f"  • Character Error Rate (CER): {raw_cer:.2f}%")
    print("-" * 65)
    print(f"AFTER ARSLAN OUTPUT CORRECTION:")
    print(f"  • Word Error Rate (WER):     {corr_wer:.2f}%")
    print(f"  • Character Error Rate (CER): {corr_cer:.2f}%")
    print("-" * 65)
    if wer_diff >= 0:
        print(f"NET GAIN:                     -{wer_diff:.2f}% Absolute WER Drop ({rel_wer_reduction:.2f}% Relative)")
        print(f"CER GAIN:                     -{cer_diff:.2f}% Absolute CER Drop")
    else:
        print(f"NET CHANGE:                   +{abs(wer_diff):.2f}% Absolute WER Change")
    print("=" * 65)

    # Print sample improvements
    print("\n[*] Sample Comparisons (Top Modifications):")
    sample_count = 0
    for r in corrected_records:
        if r["norm_raw_hypothesis"] != r["norm_corrected_hypothesis"]:
            print(f"  Key: {r.get('key', 'sample')}")
            print(f"    REF:  {r['norm_reference']}")
            print(f"    RAW:  {r['norm_raw_hypothesis']}")
            print(f"    CORR: {r['norm_corrected_hypothesis']}")
            print("    " + "-" * 50)
            sample_count += 1
            if sample_count >= 5:
                break

    with open(args.output, "w", encoding="utf-8") as f:
        for r in corrected_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[*] Corrected predictions successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
