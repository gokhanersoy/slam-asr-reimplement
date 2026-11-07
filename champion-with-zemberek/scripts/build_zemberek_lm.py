#!/usr/bin/env python3
"""Build and save a standalone Zemberek N-Gram Language Model table from a manifest or text file.

Usage:
    python scripts/build_zemberek_lm.py \
        --input data/issai_tsc/train.jsonl \
        --output zemberek_issai_lm.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slam_asr.utils.zemberek_lm import ZemberekLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Build and save Zemberek N-Gram LM")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL manifest or raw text file")
    parser.add_argument("--output", type=str, default="zemberek_lm.json", help="Output path for precomputed LM (.json)")
    parser.add_argument("--order", type=int, default=3, help="N-gram order (default: 3 for trigram)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"[*] Building Zemberek {args.order}-gram Language Model from: {args.input}")
    lm = ZemberekLanguageModel(manifest_path=args.input, ngram_order=args.order)
    print(f"[*] Indexed {len(lm.vocab):,} vocabulary terms, {len(lm.unigrams):,} unigrams, {len(lm.bigrams):,} bigrams, {len(lm.trigrams):,} trigrams.")

    lm.save_lm(args.output)
    print(f"[*] Precomputed Zemberek LM successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
