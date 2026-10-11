#!/usr/bin/env python3
"""Utility script to prepare LibriSpeech JSONL manifests for SLAM-ASR training and evaluation."""

import argparse
import glob
import json
import os
from tqdm import tqdm


def prepare_from_local_dir(librispeech_dir: str, split: str, output_jsonl: str):
    """Scan a local LibriSpeech directory structure (e.g., LibriSpeech/train-clean-100/):

    Extracts audio paths and matching transcript lines (.trans.txt).
    """
    split_dir = os.path.join(librispeech_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # LibriSpeech structure: <split>/<reader_id>/<chapter_id>/<reader_id>-<chapter_id>.trans.txt
    trans_files = glob.glob(os.path.join(split_dir, "*", "*", "*.trans.txt"))
    records = []

    print(f"[*] Processing {len(trans_files)} transcript files in {split_dir}...")
    for trans_path in tqdm(trans_files, desc=f"Scanning {split}"):
        chapter_dir = os.path.dirname(trans_path)
        with open(trans_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                key, text = parts
                # Audio can be .flac or .wav
                flac_path = os.path.join(chapter_dir, f"{key}.flac")
                wav_path = os.path.join(chapter_dir, f"{key}.wav")
                audio_path = flac_path if os.path.exists(flac_path) else wav_path

                if not os.path.exists(audio_path):
                    continue

                records.append({
                    "key": key,
                    "source": os.path.abspath(audio_path),
                    "target": text,
                })

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[*] Wrote {len(records)} samples to {output_jsonl}")


def prepare_from_huggingface(split_name: str, output_jsonl: str):
    """Download and export a LibriSpeech split from Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets via `pip install datasets` to download from Hugging Face.")

    print(f"[*] Loading '{split_name}' from Hugging Face 'librispeech_asr'...")
    ds = load_dataset("librispeech_asr", split=split_name)

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc=f"Exporting {split_name}"):
            record = {
                "key": item["id"],
                "source": item["audio"]["path"],
                "target": item["text"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"[*] Exported {len(ds)} samples to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Generate SLAM-ASR JSONL Manifests")
    parser.add_argument("--mode", choices=["local", "hf"], default="local", help="Source type: local directory or Hugging Face")
    parser.add_argument("--librispeech_dir", type=str, default=None, help="Root path of LibriSpeech dataset (for local mode)")
    parser.add_argument("--split", type=str, required=True, help="Split name (e.g. train-clean-100, dev-other, test-clean)")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL manifest path")
    args = parser.parse_args()

    if args.mode == "local":
        if not args.librispeech_dir:
            raise ValueError("--librispeech_dir is required when --mode=local")
        prepare_from_local_dir(args.librispeech_dir, args.split, args.output_jsonl)
    else:
        prepare_from_huggingface(args.split, args.output_jsonl)


if __name__ == "__main__":
    main()
