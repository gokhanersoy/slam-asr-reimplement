#!/usr/bin/env python3
"""Data preparation script for the ISSAI Turkish Speech Corpus (ISSAI_TSC_218).

Dataset structure inside ISSAI_TSC_218.tar.gz:
  - Dev/   : Paired audio and transcript files: 1.wav, 1.txt, 2.wav, 2.txt, ...
  - Test/  : Paired audio and transcript files: 1.wav, 1.txt, 2.wav, 2.txt, ...
  - Train/ : Paired audio and transcript files: 1.wav, 1.txt, 2.wav, 2.txt, ...

Parses each split directory, reads transcripts from matching .txt files,
applies accurate Turkish text normalization (I -> ı, İ -> i, ç, ğ, ö, ş, ü),
calculates audio durations, and generates JSONL manifests for SLAM-ASR:
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
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of samples per split (useful for quick prototyping / subsets)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic behavior",
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
        print("[!] 'huggingface_hub' is not installed. Installing and retrying...")
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

    print(f"[*] Extracting '{tar_path}' to '{extract_dir}' (this may take a few minutes)...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    with open(marker_file, "w") as f:
        f.write("OK\n")
    print(f"[*] Extraction complete: {extract_dir}")


def get_wav_duration(wav_path: str) -> float:
    """Read duration from WAV header directly without loading entire audio array into memory."""
    try:
        with wave.open(wav_path, "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return round(frames / float(rate), 3)
    except Exception:
        return 0.0


def locate_split_directories(extract_dir: str) -> Dict[str, str]:
    """Find Train, Dev, and Test directories inside extract_dir.

    ISSAI_TSC_218 contains:
      - Train/ (or train/)
      - Dev/   (or dev/ or val/)
      - Test/  (or test/)
    """
    splits: Dict[str, str] = {}
    for root, dirs, _ in os.walk(extract_dir):
        for d in dirs:
            d_lower = d.lower()
            if d_lower == "train" and "train" not in splits:
                splits["train"] = os.path.join(root, d)
            elif d_lower in ("dev", "val", "validation") and "val" not in splits:
                splits["val"] = os.path.join(root, d)
            elif d_lower == "test" and "test" not in splits:
                splits["test"] = os.path.join(root, d)
    return splits


def scan_paired_directory(
    dir_path: str,
    split_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, any]]:
    """Scan directory containing paired audio and transcript files:

    e.g. 1.wav, 1.txt, 2.wav, 2.txt, ...
    """
    samples: List[Dict[str, any]] = []
    if not os.path.isdir(dir_path):
        return samples

    # Collect all audio files in the folder (or subfolders)
    audio_files: List[str] = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith((".wav", ".flac")) and not f.startswith("."):
                audio_files.append(os.path.join(root, f))

    # Sort numerically if names are digits (1.wav, 2.wav), otherwise alphabetically
    def sort_key(p: str):
        stem = os.path.splitext(os.path.basename(p))[0]
        return int(stem) if stem.isdigit() else stem

    audio_files.sort(key=sort_key)
    print(f"[*] Found {len(audio_files):,d} audio files in [{split_name.upper()}] ({dir_path})")

    if max_samples is not None and len(audio_files) > max_samples:
        print(f"    --> Subsetting [{split_name.upper()}] to first {max_samples:,d} samples (--max_samples).")
        audio_files = audio_files[:max_samples]

    skipped_no_txt = 0
    skipped_empty_txt = 0

    for wav_path in audio_files:
        stem = os.path.splitext(wav_path)[0]
        txt_path = stem + ".txt"

        # Check for transcript file
        if not os.path.isfile(txt_path):
            skipped_no_txt += 1
            continue

        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as tf:
                raw_text = tf.read().strip()
        except Exception:
            skipped_empty_txt += 1
            continue

        clean_text = normalize_text_turkish(raw_text)
        if not clean_text:
            skipped_empty_txt += 1
            continue

        dur = get_wav_duration(wav_path)
        base_stem = os.path.splitext(os.path.basename(wav_path))[0]
        key = f"issai_{split_name}_{base_stem}"

        abs_audio = os.path.abspath(wav_path)
        samples.append({
            "key": key,
            "audio_filepath": abs_audio,
            "source": abs_audio,
            "text": clean_text,
            "target": clean_text,
            "duration": dur,
        })

    if skipped_no_txt > 0 or skipped_empty_txt > 0:
        print(f"    (Skipped: {skipped_no_txt} without .txt, {skipped_empty_txt} with empty text)")

    return samples


def find_tsv_fallback_files(extract_dir: str) -> List[str]:
    """Fallback: Search for TSV / CSV metadata files if paired .txt files aren't found."""
    candidates = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.endswith((".tsv", ".csv")) and not f.startswith("."):
                lower_f = f.lower()
                if any(k in lower_f for k in ["transcript", "metadata", "train", "val", "test", "dev"]):
                    candidates.append(os.path.join(root, f))
    return candidates


def parse_tsv_fallback(filepath: str, extract_dir: str) -> List[Dict[str, any]]:
    """Fallback TSV parser."""
    samples = []
    delimiter = "\t" if filepath.endswith(".tsv") else ","
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) < 2:
                continue
            raw_audio, raw_text = row[0].strip(), row[1].strip()
            if not raw_audio or not raw_text:
                continue
            audio_path = os.path.join(extract_dir, raw_audio)
            if not os.path.isfile(audio_path):
                continue
            clean_text = normalize_text_turkish(raw_text)
            dur = get_wav_duration(audio_path)
            key = os.path.splitext(os.path.basename(audio_path))[0]
            abs_p = os.path.abspath(audio_path)
            samples.append({
                "key": key,
                "audio_filepath": abs_p,
                "source": abs_p,
                "text": clean_text,
                "target": clean_text,
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

    # 3. Locate Train, Dev, and Test Directories (ISSAI_TSC_218 Structure)
    splits = locate_split_directories(extract_dir)
    print("\n[*] Located ISSAI_TSC_218 Dataset Split Directories:")
    for name in ["train", "val", "test"]:
        d_path = splits.get(name)
        if d_path:
            print(f"    • [{name.upper():5s}]: {os.path.relpath(d_path, output_dir)}")
        else:
            print(f"    • [{name.upper():5s}]: (Not found under '{extract_dir}')")

    split_samples: Dict[str, List[Dict[str, any]]] = {"train": [], "val": [], "test": []}

    # Primary Scanner: Paired wav & txt files (1.wav, 1.txt)
    for split_name in ["train", "val", "test"]:
        if split_name in splits:
            samples = scan_paired_directory(
                dir_path=splits[split_name],
                split_name=split_name,
                max_samples=args.max_samples,
            )
            split_samples[split_name] = samples

    # Fallback if paired scan produced no files
    if sum(len(s) for s in split_samples.values()) == 0:
        print("[!] No paired .wav and .txt files found in split folders. Checking for TSV/CSV metadata fallback...")
        tsv_files = find_tsv_fallback_files(extract_dir)
        for tf in tsv_files:
            b = os.path.basename(tf).lower()
            s_list = parse_tsv_fallback(tf, extract_dir)
            if "train" in b:
                split_samples["train"].extend(s_list)
            elif any(k in b for k in ["dev", "val"]):
                split_samples["val"].extend(s_list)
            elif "test" in b:
                split_samples["test"].extend(s_list)

    # 4. Save JSONL Manifests
    train_manifest = os.path.join(output_dir, "train.jsonl")
    val_manifest = os.path.join(output_dir, "val.jsonl")
    test_manifest = os.path.join(output_dir, "test.jsonl")

    manifest_map = {
        "train": (train_manifest, split_samples["train"]),
        "val": (val_manifest, split_samples["val"]),
        "test": (test_manifest, split_samples["test"]),
    }

    print("\n" + "=" * 70)
    print("[*] ISSAI TURKISH SPEECH CORPUS (ISSAI_TSC_218) MANIFEST SUMMARY")
    print("=" * 70)

    for split_name, (m_path, samples) in manifest_map.items():
        total_sec = sum(s.get("duration", 0.0) for s in samples)
        total_hours = total_sec / 3600.0

        with open(m_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"  [{split_name.upper():5s}] {len(samples):7,d} utterances | {total_hours:6.2f} hours -> {m_path}")

    print("=" * 70)
    if len(split_samples["train"]) > 0:
        sample = split_samples["train"][0]
        print(f"[*] Sample Manifest Entry:\n"
              f"    Key:      {sample['key']}\n"
              f"    Audio:    {sample['audio_filepath']}\n"
              f"    Text:     {sample['text']}\n"
              f"    Duration: {sample['duration']}s\n")
    print("[*] ISSAI Turkish Speech Corpus manifests prepared successfully.")


if __name__ == "__main__":
    main()
