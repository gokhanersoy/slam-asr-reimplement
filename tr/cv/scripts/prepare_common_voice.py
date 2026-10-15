#!/usr/bin/env python3
"""Data preparation script for Mozilla Common Voice (Turkish) via Mozilla Data Collective (MDC).

Dataset ID: cmu5wkah500c2o10719lem8m8
API Endpoint: https://mozilladatacollective.com/api/datasets/cmu5wkah500c2o10719lem8m8/download

Downloads or extracts Common Voice Turkish dataset, parses train.tsv, dev.tsv, test.tsv,
applies Turkish text normalization (İ -> i, I -> ı, preserving ç, ğ, ö, ş, ü),
and creates JSONL manifests for SLAM-ASR:
  - data/cv/train.jsonl
  - data/cv/val.jsonl
  - data/cv/test.jsonl
"""

import argparse
import csv
import json
import os
import re
import sys
import tarfile
import urllib.request
import urllib.error
import wave
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import normalize_text_turkish
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slam_asr.utils.metrics import normalize_text_turkish

MDC_DATASET_ID = "cmu5wkah500c2o10719lem8m8"
MDC_API_URL = f"https://mozilladatacollective.com/api/datasets/{MDC_DATASET_ID}/download"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Mozilla Common Voice Turkish for SLAM-ASR")
    parser.add_argument(
        "--tar_path",
        type=str,
        default=None,
        help="Path to an already downloaded Common Voice archive (.tar.gz / .tar)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset archive from Mozilla Data Collective API",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Mozilla Data Collective API Key (or set MDC_API_KEY environment variable)",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=MDC_DATASET_ID,
        help="Mozilla Data Collective Dataset ID (default: cmu5wkah500c2o10719lem8m8)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cv",
        help="Directory to store downloaded archive, extracted files, and JSONL manifests",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of samples per split (useful for quick prototyping / subsets)",
    )
    return parser.parse_args()


def get_api_key(cli_key: Optional[str]) -> Optional[str]:
    """Retrieve API key from CLI argument, environment variables, or Google Colab userdata."""
    if cli_key:
        return cli_key.strip()

    # Check environment variables
    for env_var in ["MDC_API_KEY", "COMMON_VOICE_API_KEY"]:
        if os.environ.get(env_var):
            return os.environ.get(env_var).strip()

    # Check Google Colab secret (userdata)
    try:
        from google.colab import userdata
        key = userdata.get("MDC_API_KEY")
        if key:
            return key.strip()
    except Exception:
        pass

    return None


def fetch_download_url_from_mdc(dataset_id: str, api_key: str) -> Tuple[str, str]:
    """Query MDC API to get presigned download URL and filename."""
    url = f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "MozillaDataCollectiveClient/1.0",
    }

    print(f"[*] Requesting presigned download URL from Mozilla Data Collective API...")
    print(f"    URL: {url}")

    data = None
    # 1. Try urllib POST with empty body or empty JSON
    for body_data in [b"{}", b""]:
        try:
            req = urllib.request.Request(url, data=body_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PermissionError(
                    f"MDC API Authentication failed (HTTP {e.code}). Please ensure your MDC_API_KEY is valid "
                    f"and you have accepted the dataset terms at https://mozilladatacollective.com"
                ) from e
            continue
        except Exception:
            continue

    # 2. If urllib failed, try curl as a robust fallback (exactly matching MDC docs)
    if data is None:
        try:
            import subprocess
            curl_cmd = [
                "curl", "-sS", "-X", "POST", url,
                "-H", f"Authorization: Bearer {api_key}",
                "-H", "Content-Type: application/json",
            ]
            res = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
        except Exception:
            pass

    if data is None:
        raise RuntimeError(f"Could not retrieve download URL from MDC API endpoint: {url}")

    download_url = data.get("downloadUrl") or data.get("url")
    filename = data.get("filename") or f"common_voice_{dataset_id}.tar.gz"

    if not download_url:
        raise ValueError(f"MDC API response did not contain 'downloadUrl'. Response: {data}")

    return download_url, filename


def download_file_with_progress(url: str, dest_path: str):
    """Download file with chunking and progress logging."""
    print(f"[*] Downloading dataset to: {dest_path}")
    temp_path = dest_path + ".tmp"
    req = urllib.request.Request(url, headers={"User-Agent": "MozillaDataCollectiveClient/1.0"})

    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8 * 1024 * 1024  # 8 MB chunks
            last_percent = -1

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = int(downloaded * 100 / total_size)
                    if percent != last_percent and percent % 5 == 0:
                        last_percent = percent
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"    Progress: {percent}% ({mb_down:.1f} MB / {mb_total:.1f} MB)")
                else:
                    mb_down = downloaded / (1024 * 1024)
                    if int(mb_down) % 50 == 0:
                        print(f"    Downloaded: {mb_down:.1f} MB")

        os.replace(temp_path, dest_path)
        print(f"[*] Download completed successfully: {dest_path}")
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise e


def download_from_mdc(output_dir: str, dataset_id: str, api_key: str) -> str:
    """Download Common Voice archive via direct REST API (recommended) or datacollective SDK."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Primary Method: Direct REST API (Streaming with progress bar and exact destination)
    try:
        print("[*] Connecting to Mozilla Data Collective REST API...")
        download_url, filename = fetch_download_url_from_mdc(dataset_id, api_key)
        dest_path = os.path.join(output_dir, filename)

        if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
            print(f"[*] Archive already exists at: {dest_path}")
            return dest_path

        download_file_with_progress(download_url, dest_path)
        return dest_path
    except Exception as rest_err:
        print(f"[!] REST API download encountered an issue: {rest_err}")

    # 2. Secondary Fallback: Official datacollective SDK if installed
    try:
        import datacollective
        print("[*] Attempting fallback via official 'datacollective' SDK...")
        os.environ["MDC_API_KEY"] = api_key
        tar_path = datacollective.download_dataset(dataset_id)
        print(f"[*] Download completed via SDK to: {tar_path}")
        return tar_path
    except Exception as sdk_err:
        raise RuntimeError(
            f"Failed to download Common Voice dataset from MDC.\n"
            f"  • REST API error: {rest_err}\n"
            f"  • SDK error: {sdk_err}\n"
            f"Please verify your MDC_API_KEY and ensure you accepted the dataset terms on https://mozilladatacollective.com"
        ) from rest_err


def extract_archive(tar_path: str, extract_dir: str):
    """Extract tar.gz or tar archive."""
    os.makedirs(extract_dir, exist_ok=True)
    marker_file = os.path.join(extract_dir, ".extracted_marker")
    if os.path.exists(marker_file):
        print(f"[*] Archive already extracted in: {extract_dir}")
        return

    print(f"[*] Extracting '{tar_path}' to '{extract_dir}' (this may take a few minutes)...")
    mode = "r:gz" if tar_path.endswith((".gz", ".tgz")) else "r:"
    with tarfile.open(tar_path, mode) as tar:
        tar.extractall(path=extract_dir)

    with open(marker_file, "w") as f:
        f.write("OK\n")
    print(f"[*] Extraction complete: {extract_dir}")


def get_audio_duration(audio_path: str) -> float:
    """Determine audio duration for WAV or MP3 files."""
    if audio_path.endswith(".wav"):
        try:
            with wave.open(audio_path, "rb") as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return round(frames / float(rate), 3)
        except Exception:
            return 0.0

    # Try torchaudio for MP3 if installed
    try:
        import torchaudio
        info = torchaudio.info(audio_path)
        return round(info.num_frames / float(info.sample_rate), 3)
    except Exception:
        pass

    # Try mutagen for MP3 if installed
    try:
        from mutagen.mp3 import MP3
        audio = MP3(audio_path)
        return round(audio.info.length, 3)
    except Exception:
        pass

    # Estimate based on file size if header parser unavailable (~48kbps MP3 default)
    try:
        size_bytes = os.path.getsize(audio_path)
        return round(size_bytes / 6000.0, 3)  # Rough heuristic fallback
    except Exception:
        return 0.0


def locate_common_voice_files(extract_dir: str) -> Tuple[Dict[str, str], str]:
    """Find Common Voice TSV files (train.tsv, dev.tsv, test.tsv) and clips directory."""
    tsv_files: Dict[str, str] = {}
    clips_dir: Optional[str] = None

    for root, dirs, files in os.walk(extract_dir):
        # Look for clips folder
        for d in dirs:
            if d.lower() == "clips":
                clips_dir = os.path.join(root, d)

        # Look for split TSVs
        for f in files:
            f_lower = f.lower()
            if f_lower == "train.tsv":
                tsv_files["train"] = os.path.join(root, f)
            elif f_lower in ("dev.tsv", "val.tsv", "validation.tsv"):
                tsv_files["val"] = os.path.join(root, f)
            elif f_lower == "test.tsv":
                tsv_files["test"] = os.path.join(root, f)

    if clips_dir is None:
        # Fallback: check if any directory contains mp3 files
        for root, _, files in os.walk(extract_dir):
            if any(f.endswith((".mp3", ".wav")) for f in files):
                clips_dir = root
                break

    return tsv_files, clips_dir


def parse_common_voice_tsv(
    tsv_path: str,
    clips_dir: str,
    split_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, any]]:
    """Parse Common Voice TSV file and match clips."""
    samples = []
    if not os.path.isfile(tsv_path):
        return samples

    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return samples

        path_idx = 1
        sentence_idx = 2
        for idx, col in enumerate(header):
            col_l = col.strip().lower()
            if col_l == "path":
                path_idx = idx
            elif col_l == "sentence":
                sentence_idx = idx

        for row in reader:
            if len(row) <= max(path_idx, sentence_idx):
                continue

            raw_path = row[path_idx].strip()
            raw_sentence = row[sentence_idx].strip()

            if not raw_path or not raw_sentence:
                continue

            # Check audio file existence
            audio_path = os.path.join(clips_dir, raw_path)
            if not os.path.isfile(audio_path):
                # Check without/with extension
                if not raw_path.endswith((".mp3", ".wav")):
                    for ext in [".mp3", ".wav"]:
                        cand = os.path.join(clips_dir, raw_path + ext)
                        if os.path.isfile(cand):
                            audio_path = cand
                            break

            if not os.path.isfile(audio_path):
                continue

            clean_text = normalize_text_turkish(raw_sentence)
            if not clean_text:
                continue

            dur = get_audio_duration(audio_path)
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            key = f"cv_{split_name}_{stem}"

            abs_audio = os.path.abspath(audio_path)
            samples.append({
                "key": key,
                "audio_filepath": abs_audio,
                "source": abs_audio,
                "text": clean_text,
                "target": clean_text,
                "duration": dur,
            })

            if max_samples and len(samples) >= max_samples:
                break

    return samples


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Determine Tar Archive Path
    tar_path = args.tar_path
    if tar_path is None:
        api_key = get_api_key(args.api_key)
        if args.download or api_key:
            if not api_key:
                print(
                    "[!] Error: No API Key found!\n"
                    "    Please supply your Mozilla Data Collective API key via:\n"
                    "      • --api_key 'YOUR_KEY'\n"
                    "      • Or set environment variable: export MDC_API_KEY='YOUR_KEY'\n"
                    "      • Or add 'MDC_API_KEY' to Colab Secrets (userdata)"
                )
                sys.exit(1)
            tar_path = download_from_mdc(output_dir, args.dataset_id, api_key)
        else:
            # Check if any .tar.gz or .tar is already in output_dir
            for f in os.listdir(output_dir):
                if f.endswith((".tar.gz", ".tar", ".tgz")):
                    tar_path = os.path.join(output_dir, f)
                    print(f"[*] Found existing archive in output_dir: {tar_path}")
                    break

            if not tar_path:
                print(
                    "[!] No --tar_path provided and no archive found in output directory.\n"
                    "    To download from Mozilla Data Collective, run with:\n"
                    f"      python scripts/prepare_common_voice.py --download --api_key '<YOUR_MDC_API_KEY>'\n"
                    "    Or provide a local file path:\n"
                    "      python scripts/prepare_common_voice.py --tar_path /path/to/common_voice.tar.gz"
                )
                sys.exit(1)

    # 2. Extract Archive
    extract_dir = os.path.join(output_dir, "extracted")
    extract_archive(tar_path, extract_dir)

    # 3. Locate TSV Files and Clips Folder
    tsv_files, clips_dir = locate_common_voice_files(extract_dir)
    print("\n[*] Located Common Voice Turkish Files:")
    print(f"    • Clips Directory: {os.path.relpath(clips_dir, output_dir) if clips_dir else 'NOT FOUND'}")
    for name in ["train", "val", "test"]:
        p = tsv_files.get(name)
        print(f"    • [{name.upper():5s}] TSV:      {os.path.relpath(p, output_dir) if p else 'NOT FOUND'}")

    if not clips_dir or not os.path.isdir(clips_dir):
        print(f"[!] Error: Could not locate 'clips/' audio directory under {extract_dir}")
        sys.exit(1)

    # 4. Parse Each Split
    split_samples: Dict[str, List[Dict[str, any]]] = {"train": [], "val": [], "test": []}
    for split_name in ["train", "val", "test"]:
        tsv_path = tsv_files.get(split_name)
        if tsv_path:
            print(f"[*] Parsing [{split_name.upper()}] from: {os.path.basename(tsv_path)}...")
            samples = parse_common_voice_tsv(
                tsv_path=tsv_path,
                clips_dir=clips_dir,
                split_name=split_name,
                max_samples=args.max_samples,
            )
            split_samples[split_name] = samples

    # 5. Save JSONL Manifests
    train_manifest = os.path.join(output_dir, "train.jsonl")
    val_manifest = os.path.join(output_dir, "val.jsonl")
    test_manifest = os.path.join(output_dir, "test.jsonl")

    manifest_map = {
        "train": (train_manifest, split_samples["train"]),
        "val": (val_manifest, split_samples["val"]),
        "test": (test_manifest, split_samples["test"]),
    }

    print("\n" + "=" * 70)
    print("[*] MOZILLA COMMON VOICE (TURKISH) MANIFEST SUMMARY")
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
    print("[*] Common Voice Turkish manifests prepared successfully.")


if __name__ == "__main__":
    main()
