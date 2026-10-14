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
        # Also check if split exists directly without nesting
        matches = glob.glob(os.path.join(librispeech_dir, "**", split), recursive=True)
        if matches:
            split_dir = matches[0]
        else:
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

    trans_files = glob.glob(os.path.join(split_dir, "**", "*.trans.txt"), recursive=True)
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
                flac_path = os.path.join(chapter_dir, f"{key}.flac")
                wav_path = os.path.join(chapter_dir, f"{key}.wav")
                audio_path = flac_path if os.path.exists(flac_path) else wav_path

                if not os.path.exists(audio_path):
                    continue

                records.append({
                    "key": key,
                    "source": os.path.abspath(audio_path),
                    "target": text,
                    "prompt": "Transcribe speech to text.",
                })

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[*] Wrote {len(records)} samples to {output_jsonl}")


def prepare_from_huggingface(split_name: str, output_jsonl: str, max_samples: int = 500):
    """Download and export a LibriSpeech split from Hugging Face using namespaced dataset openslr/librispeech_asr."""
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install datasets and soundfile: pip install datasets soundfile")

    print(f"[*] Loading '{split_name}' from Hugging Face 'openslr/librispeech_asr'...")
    # Use canonical namespace: openslr/librispeech_asr
    try:
        ds = load_dataset("openslr/librispeech_asr", "clean", split=split_name, streaming=True)
    except Exception as e:
        print(f"[!] Warning: Falling back to distil-whisper/librispeech_asr: {e}")
        ds = load_dataset("distil-whisper/librispeech_asr", "clean", split=split_name, streaming=True)

    audio_dir = os.path.join(os.path.dirname(os.path.abspath(output_jsonl)), "hf_audio")
    os.makedirs(audio_dir, exist_ok=True)

    records = []
    count = 0
    for item in tqdm(ds, desc=f"Exporting {split_name}", total=max_samples):
        if max_samples and count >= max_samples:
            break

        key = item.get("id") or item.get("chapter_id", f"sample_{count}")
        text = item.get("text", "")
        audio_info = item["audio"]

        wav_path = os.path.join(audio_dir, f"{key}.wav")
        if not os.path.exists(wav_path):
            sf.write(wav_path, audio_info["array"], audio_info["sampling_rate"])

        records.append({
            "key": str(key),
            "source": os.path.abspath(wav_path),
            "target": text,
            "prompt": "Transcribe speech to text.",
        })
        count += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[*] Exported {len(records)} samples to {output_jsonl}")


def prepare_from_torchaudio(url: str, root_dir: str, output_jsonl: str):
    """Download directly using torchaudio.datasets.LIBRISPEECH (reliable OpenSLR mirror)."""
    try:
        import torchaudio
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    print(f"[*] Downloading '{url}' via torchaudio...")
    dataset = torchaudio.datasets.LIBRISPEECH(root=root_dir, url=url, download=True)
    librispeech_dir = os.path.join(root_dir, "LibriSpeech")
    prepare_from_local_dir(librispeech_dir, url, output_jsonl)


def main():
    parser = argparse.ArgumentParser(description="Generate SLAM-ASR JSONL Manifests")
    parser.add_argument("--mode", choices=["local", "hf", "torchaudio"], default="local")
    parser.add_argument("--librispeech_dir", type=str, default="./data", help="Root directory")
    parser.add_argument("--split", type=str, default="validation", help="Split name (e.g. train.100, validation, dev-clean)")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL manifest path")
    parser.add_argument("--max_samples", type=int, default=500, help="Max samples for HF mode")
    args = parser.parse_args()

    if args.mode == "local":
        prepare_from_local_dir(args.librispeech_dir, args.split, args.output_jsonl)
    elif args.mode == "hf":
        prepare_from_huggingface(args.split, args.output_jsonl, max_samples=args.max_samples)
    elif args.mode == "torchaudio":
        prepare_from_torchaudio(args.split, args.librispeech_dir, args.output_jsonl)


if __name__ == "__main__":
    main()
