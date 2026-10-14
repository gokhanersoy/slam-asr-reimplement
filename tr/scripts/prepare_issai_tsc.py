#!/usr/bin/env python3
"""Data preparation script for the ISSAI Turkish Speech Corpus (ISSAI_TSC_218).

Downloads or extracts ISSAI_TSC_218.tar.gz from Hugging Face:
  https://huggingface.co/datasets/issai/Turkish_Speech_Corpus

Parses transcripts, verifies audio files, applies Turkish text normalization,
and generates JSONL manifests for SLAM-ASR training, validation, and evaluation:
  - data/issai_tsc/train.jsonl
  - data/issai_tsc/val.jsonl
  - data/issai_tsc/test.jsonl
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import tarfile
import wave
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import normalize_text_turkish
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slam_asr.utils.metrics import normalize_text_turkish


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ISSAI Turkish Speech Corpus for SLAM-ASR")
    parser.add_argument(
        "--tar_path",
        type=str,
        default=None,
        help="Path to existing ISSAI_TSC_218.tar.gz archive. If None, can download via --download",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download ISSAI_TSC_218.tar.gz directly from Hugging Face Hub (issai/Turkish_Speech_Corpus)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/issai_tsc",
        help="Directory to extract audio files and write JSONL manifests",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Optional Hugging Face access token for faster/authenticated download",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.90,
        help="Ratio of data assigned to train set (default: 0.90)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Ratio of data assigned to val set (default: 0.05)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Ratio of data assigned to test set (default: 0.05)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to process (useful for quick prototyping)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val/test splitting",
    )
    return parser.parse_args()


def download_from_hf(output_dir: str, token: Optional[str] = None) -> str:
    """Download ISSAI_TSC_218.tar.gz from Hugging Face Hub."""
    print("[*] Downloading ISSAI_TSC_218.tar.gz from huggingface.co/datasets/issai/Turkish_Speech_Corpus...")
    try:
        from huggingface_hub import hf_hub_download
        tar_path = hf_hub_download(
            repo_id="issai/Turkish_Speech_Corpus",
            filename="ISSAI_TSC_218.tar.gz",
            repo_type="dataset",
            token=token,
            local_dir=output_dir,
        )
        print(f"[*] Download completed: {tar_path}")
        return tar_path
    except ImportError:
        print("[!] 'huggingface_hub' is not installed. Installing or using curl fallback...")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import hf_hub_download
        tar_path = hf_hub_download(
            repo_id="issai/Turkish_Speech_Corpus",
            filename="ISSAI_TSC_218.tar.gz",
            repo_type="dataset",
            token=token,
            local_dir=output_dir,
        )
        return tar_path


def extract_archive(tar_path: str, extract_dir: str):
    """Extract tar.gz archive if not already extracted."""
    os.makedirs(extract_dir, exist_ok=True)
    marker_file = os.path.join(extract_dir, ".extracted_marker")
    if os.path.exists(marker_file):
        print(f"[*] Archive already extracted in: {extract_dir}")
        return

    print(f"[*] Extracting '{tar_path}' to '{extract_dir}'...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    with open(marker_file, "w") as f:
        f.write("OK\n")
    print(f"[*] Extraction complete: {extract_dir}")


def get_wav_duration(wav_path: str) -> float:
    """Read duration from WAV header directly without loading entire audio array."""
    try:
        with wave.open(wav_path, "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return round(frames / float(rate), 3)
    except Exception:
        return 0.0


def find_metadata_files(extract_dir: str) -> List[str]:
    """Search for TSV / CSV / TXT transcript files inside the extracted directory."""
    candidates = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.endswith((".tsv", ".csv", ".txt")) and not f.startswith("."):
                lower_f = f.lower()
                if any(k in lower_f for k in ["transcript", "metadata", "train", "val", "test", "dev", "data"]):
                    candidates.append(os.path.join(root, f))
    return candidates


def parse_transcript_file(filepath: str, extract_dir: str) -> List[Dict[str, any]]:
    """Parse TSV or CSV transcript file and link to audio files."""
    samples = []
    delimiter = "\t" if filepath.endswith(".tsv") else ("," if filepath.endswith(".csv") else "\t")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        # Detect whether first line is a header
        first_line = f.readline()
        f.seek(0)
        has_header = any(h in first_line.lower() for h in ["audio", "path", "file", "text", "transcript", "sentence"])

        reader = csv.reader(f, delimiter=delimiter)
        audio_col_idx = 0
        text_col_idx = 1

        if has_header:
            header = [h.strip().lower() for h in next(reader)]
            for idx, col in enumerate(header):
                if any(k in col for k in ["audio", "path", "file", "wav"]):
                    audio_col_idx = idx
                elif any(k in col for k in ["text", "transcript", "sentence", "label"]):
                    text_col_idx = idx

        for row_idx, row in enumerate(reader):
            if not row or len(row) <= max(audio_col_idx, text_col_idx):
                continue
            raw_audio = row[audio_col_idx].strip()
            raw_text = row[text_col_idx].strip()
            if not raw_audio or not raw_text:
                continue

            # Resolve audio file location
            audio_path = None
            candidate_paths = [
                raw_audio,
                os.path.join(os.path.dirname(filepath), raw_audio),
                os.path.join(extract_dir, raw_audio),
                os.path.join(extract_dir, "Audios", raw_audio),
                os.path.join(extract_dir, "audios", raw_audio),
                os.path.join(extract_dir, "Wav", raw_audio),
                os.path.join(extract_dir, "wav", raw_audio),
            ]
            if not raw_audio.endswith((".wav", ".flac", ".mp3")):
                candidate_paths.extend([p + ".wav" for p in candidate_paths])

            for p in candidate_paths:
                if os.path.isfile(p):
                    audio_path = os.path.abspath(p)
                    break

            if audio_path is None:
                continue

            clean_text = normalize_text_turkish(raw_text)
            if not clean_text:
                continue

            dur = get_wav_duration(audio_path)
            key = os.path.splitext(os.path.basename(audio_path))[0]

            samples.append({
                "key": key,
                "audio_filepath": audio_path,
                "text": clean_text,
                "duration": dur,
            })

    return samples


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Acquire Tar Archive
    tar_path = args.tar_path
    if tar_path is None:
        if args.download:
            tar_path = download_from_hf(output_dir, token=args.hf_token)
        else:
            default_candidate = os.path.join(output_dir, "ISSAI_TSC_218.tar.gz")
            if os.path.isfile(default_candidate):
                tar_path = default_candidate
            else:
                print(
                    f"[!] No --tar_path provided and '{default_candidate}' not found.\n"
                    f"    To download directly from Hugging Face, run with: --download\n"
                    f"    Or specify path to local archive with: --tar_path /path/to/ISSAI_TSC_218.tar.gz"
                )
                sys.exit(1)

    # 2. Extract Archive
    extract_dir = os.path.join(output_dir, "extracted")
    extract_archive(tar_path, extract_dir)

    # 3. Locate and Parse Metadata
    meta_files = find_metadata_files(extract_dir)
    print(f"[*] Found {len(meta_files)} transcript/metadata candidate file(s):")
    for mf in meta_files:
        print(f"    • {os.path.relpath(mf, output_dir)}")

    all_samples = []
    # If standard split files exist (train.tsv, dev.tsv, test.tsv)
    split_samples = {"train": [], "val": [], "test": []}
    has_predefined_splits = False

    for mf in meta_files:
        base_mf = os.path.basename(mf).lower()
        if "train" in base_mf:
            s = parse_transcript_file(mf, extract_dir)
            split_samples["train"].extend(s)
            has_predefined_splits = True
        elif any(k in base_mf for k in ["dev", "val"]):
            s = parse_transcript_file(mf, extract_dir)
            split_samples["val"].extend(s)
            has_predefined_splits = True
        elif "test" in base_mf:
            s = parse_transcript_file(mf, extract_dir)
            split_samples["test"].extend(s)
            has_predefined_splits = True
        else:
            s = parse_transcript_file(mf, extract_dir)
            all_samples.extend(s)

    if not has_predefined_splits or (len(split_samples["train"]) == 0 and len(all_samples) > 0):
        # Create deterministic split from all_samples
        print(f"[*] Creating {int(args.train_ratio*100)}/{int(args.val_ratio*100)}/{int(args.test_ratio*100)} split from {len(all_samples)} samples...")
        random.shuffle(all_samples)

        if args.max_samples and len(all_samples) > args.max_samples:
            print(f"[*] Limiting total samples to --max_samples={args.max_samples}")
            all_samples = all_samples[:args.max_samples]

        n_total = len(all_samples)
        n_train = int(n_total * args.train_ratio)
        n_val = int(n_total * args.val_ratio)

        split_samples["train"] = all_samples[:n_train]
        split_samples["val"] = all_samples[n_train : n_train + n_val]
        split_samples["test"] = all_samples[n_train + n_val :]

    # 4. Save JSONL Manifests
    train_manifest = os.path.join(output_dir, "train.jsonl")
    val_manifest = os.path.join(output_dir, "val.jsonl")
    test_manifest = os.path.join(output_dir, "test.jsonl")

    manifest_map = {
        "train": (train_manifest, split_samples["train"]),
        "val": (val_manifest, split_samples["val"]),
        "test": (test_manifest, split_samples["test"]),
    }

    print("\n" + "=" * 65)
    print("[*] ISSAI TURKISH SPEECH CORPUS MANIFEST SUMMARY")
    print("=" * 65)

    for split_name, (m_path, samples) in manifest_map.items():
        total_sec = sum(s.get("duration", 0.0) for s in samples)
        total_hours = total_sec / 3600.0

        with open(m_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"  [{split_name.upper():5s}] {len(samples):7,d} utterances | {total_hours:6.2f} hours -> {m_path}")

    print("=" * 65)
    if len(split_samples["train"]) > 0:
        sample = split_samples["train"][0]
        print(f"[*] Sample Manifest Entry:\n    Key:      {sample['key']}\n    Audio:    {sample['audio_filepath']}\n    Text:     {sample['text']}\n    Duration: {sample['duration']}s\n")
    print("[*] Preparation completed successfully.")


if __name__ == "__main__":
    main()
