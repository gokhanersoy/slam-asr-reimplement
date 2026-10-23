#!/usr/bin/env python3
"""Evaluate saved SLAM-ASR checkpoints on Mozilla Common Voice test set locally on Desktop.

Directly reads and writes to Google Drive synced folder:
- Windows: G: drive (e.g. G:/My Drive/checkpoints or G:/MyDrive/checkpoints)
- macOS: ~/Library/CloudStorage/GoogleDrive-*/My Drive/checkpoints or /Volumes/GoogleDrive/
- Google Colab: /content/drive/MyDrive/checkpoints

Supports running on:
- NVIDIA GPU (CUDA)
- Apple Silicon Mac (MPS / M1/M2/M3/M4)
- CPU (Fallback)

Zero Colab GPU compute units or cloud subscription required.
"""

import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys


def find_default_checkpoints_dir():
    """Automatically resolve the Google Drive checkpoints folder across OS platforms (including Turkish 'Drive\'ım')."""
    candidates = []

    # 1. Windows: G: drive Google Drive mapping (Primary: G:\Drive'ım\_Akademik_\Doktora (Gazi)\checkpoints)
    for p in [
        r"G:\Drive'ım\_Akademik_\Doktora (Gazi)\checkpoints",
        "G:/Drive'ım/_Akademik_/Doktora (Gazi)/checkpoints",
        "G:/Drive'im/_Akademik_/Doktora (Gazi)/checkpoints",
        r"G:\Drive'im\_Akademik_\Doktora (Gazi)\checkpoints",
        "G:/Drive'ım/checkpoints",
        "G:\\Drive'ım\\checkpoints",
        "G:/Drive'im/checkpoints",
        "G:/My Drive/checkpoints",
        "G:\\My Drive\\checkpoints",
        "G:/MyDrive/checkpoints",
        "G:\\MyDrive\\checkpoints",
        "G:/checkpoints",
        "G:\\checkpoints",
    ]:
        candidates.append(p)

    # 2. macOS: CloudStorage (Google Drive for desktop - Drive'ım / My Drive)
    cloud_storage = os.path.expanduser("~/Library/CloudStorage")
    if os.path.exists(cloud_storage):
        for entry in sorted(glob.glob(os.path.join(cloud_storage, "GoogleDrive*"))):
            candidates.append(os.path.join(entry, "Drive'ım", "_Akademik_", "Doktora (Gazi)", "checkpoints"))
            candidates.append(os.path.join(entry, "Drive'im", "_Akademik_", "Doktora (Gazi)", "checkpoints"))
            candidates.append(os.path.join(entry, "My Drive", "_Akademik_", "Doktora (Gazi)", "checkpoints"))
            candidates.append(os.path.join(entry, "Drive'ım", "checkpoints"))
            candidates.append(os.path.join(entry, "Drive'im", "checkpoints"))
            candidates.append(os.path.join(entry, "My Drive", "checkpoints"))
            candidates.append(os.path.join(entry, "MyDrive", "checkpoints"))
            candidates.append(os.path.join(entry, "checkpoints"))

    # macOS legacy mounts & home folders
    candidates.append(os.path.expanduser("~/Google Drive/Drive'ım/_Akademik_/Doktora (Gazi)/checkpoints"))
    candidates.append(os.path.expanduser("~/Google Drive/My Drive/_Akademik_/Doktora (Gazi)/checkpoints"))
    candidates.append(os.path.expanduser("~/Google Drive/Drive'ım/checkpoints"))
    candidates.append(os.path.expanduser("~/Google Drive/My Drive/checkpoints"))
    candidates.append(os.path.expanduser("~/Google Drive/checkpoints"))
    candidates.append("/Volumes/GoogleDrive/Drive'ım/_Akademik_/Doktora (Gazi)/checkpoints")
    candidates.append("/Volumes/GoogleDrive/Drive'ım/checkpoints")
    candidates.append("/Volumes/GoogleDrive/My Drive/checkpoints")
    candidates.append("/Volumes/GoogleDrive/MyDrive/checkpoints")
    candidates.append("/Volumes/GoogleDrive/checkpoints")

    # 3. Google Colab / Linux
    candidates.append("/content/drive/MyDrive/_Akademik_/Doktora (Gazi)/checkpoints")
    candidates.append("/content/drive/Drive'ım/_Akademik_/Doktora (Gazi)/checkpoints")
    candidates.append("/content/drive/MyDrive/checkpoints")
    candidates.append("/content/drive/My Drive/checkpoints")
    candidates.append("/content/drive/Drive'ım/checkpoints")

    # Return the first candidate that actually exists and has checkpoints
    for cand in candidates:
        if os.path.exists(cand):
            # Check if it has any checkpoint subfolder
            if glob.glob(os.path.join(cand, "*slam_asr_tr_tsc*")) or glob.glob(os.path.join(cand, "*", "best_checkpoint")):
                return cand

    # If none found with contents, return first existing candidate folder
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    # Default platform fallback
    system = platform.system()
    if system == "Windows":
        for cand in [
            r"G:\Drive'ım\_Akademik_\Doktora (Gazi)\checkpoints",
            "G:/Drive'ım/_Akademik_/Doktora (Gazi)/checkpoints",
            "G:/Drive'ım/checkpoints",
            "G:/checkpoints",
        ]:
            if os.path.exists(cand):
                return cand
        return r"G:\Drive'ım\_Akademik_\Doktora (Gazi)\checkpoints"
    elif system == "Darwin":
        matches = glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive*"))
        if matches:
            for sub in ["Drive'ım/_Akademik_/Doktora (Gazi)", "Drive'ım", "My Drive", "MyDrive"]:
                cand = os.path.join(matches[0], sub, "checkpoints")
                if os.path.exists(cand):
                    return cand
            return os.path.join(matches[0], "Drive'ım", "_Akademik_", "Doktora (Gazi)", "checkpoints")
        return os.path.expanduser("~/Google Drive/Drive'ım/_Akademik_/Doktora (Gazi)/checkpoints")
    else:
        return "/content/drive/MyDrive/checkpoints"


def parse_args():
    default_ckpt_dir = find_default_checkpoints_dir()
    parser = argparse.ArgumentParser(
        description="Evaluate SLAM-ASR Checkpoints on Mozilla Common Voice Locally"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=default_ckpt_dir,
        help=f"Path to Google Drive checkpoints folder (default: {default_ckpt_dir})",
    )
    parser.add_argument(
        "--cv_manifest",
        type=str,
        default="data/cv/test.jsonl",
        help="Path to Common Voice test manifest (data/cv/test.jsonl)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model name filter (e.g. 't1_mlp_lora,t1_tail_aug_lora'). Default: all missing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cuda', 'mps', 'cpu'). Default: auto-detect",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation. Default: auto-configured by hardware",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=4,
        help="Beam size for decoding (default: 4)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip checkpoints that already have eval_results_cv_generalization.jsonl",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if eval_results_cv_generalization.jsonl exists",
    )
    return parser.parse_args()


def is_valid_manifest(path):
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                audio_path = data.get("audio_filepath", "")
                return bool(audio_path and os.path.exists(audio_path))
    except Exception:
        return False
    return False


def find_checkpoints(root_dir, model_filter=None):
    """Find all valid SLAM-ASR checkpoint directories."""
    candidates = sorted(
        glob.glob(os.path.join(root_dir, "*slam_asr_tr_tsc*", "best_checkpoint"))
        + glob.glob(os.path.join(root_dir, "*", "best_checkpoint")),
        key=lambda p: (re.search(r"(\d{8}_\d{6})", p).group(1) if re.search(r"(\d{8}_\d{6})", p) else "", p)
    )

    unique_ckpts = []
    seen_dirs = set()
    for ckpt in candidates:
        parent = os.path.dirname(ckpt.rstrip("/"))
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)

        # Exclude 'Eski' / 'eski' subdirectories
        norm_parent = parent.replace("\\", "/").lower()
        if "/eski/" in norm_parent or norm_parent.endswith("/eski") or os.path.basename(parent).lower() == "eski":
            continue

        folder_name = os.path.basename(parent)
        if model_filter:
            filters = [f.strip().lower() for f in model_filter.split(",")]
            if not any(f in folder_name.lower() for f in filters):
                continue
        unique_ckpts.append((folder_name, parent, ckpt))

    return unique_ckpts


def get_existing_metrics(eval_file):
    """Extract WER and CER from an existing eval_results file or cached summary."""
    if not os.path.exists(eval_file) or os.path.getsize(eval_file) == 0:
        return None, None

    parent_dir = os.path.dirname(eval_file)
    summary_file = os.path.join(parent_dir, "eval_results_cv_summary.json")

    # 1. Önceden oluşturulmuş hızlı özet dosyası varsa oku
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                w = d.get("wer")
                c = d.get("cer")
                if w is not None and c is not None:
                    return float(w), float(c)
        except Exception:
            pass

    # 2. JSONL dosyasını incele
    hyps, refs = [], []
    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                res = json.loads(line)
                # Tekil özet satırı formatı kontrolü
                if "metrics" in res or ("wer" in res and "hypothesis" not in res):
                    wer = res.get("metrics", {}).get("wer") or res.get("wer")
                    cer = res.get("metrics", {}).get("cer") or res.get("cer")
                    if wer is not None and cer is not None:
                        w = wer * 100 if wer <= 1.0 else wer
                        c = cer * 100 if cer <= 1.0 else cer
                        return float(w), float(c)
                hyps.append(res.get("hypothesis", ""))
                refs.append(res.get("reference", ""))

        if hyps and refs:
            for cand in [
                "tr/tsc/t5-ng-ta-gormez-tail-aug-lora",
                "tr/tsc/t0",
                "../tr/tsc/t5-ng-ta-gormez-tail-aug-lora",
            ]:
                if os.path.exists(os.path.join(cand, "slam_asr")):
                    if cand not in sys.path:
                        sys.path.insert(0, cand)
                    break

            try:
                from slam_asr.utils.metrics import compute_wer, compute_cer
                wer = compute_wer(hyps, refs, language="tr")
                cer = compute_cer(hyps, refs, language="tr")
                w = wer * 100 if wer <= 1.0 else wer
                c = cer * 100 if cer <= 1.0 else cer

                # Sonraki okumaları anlık kılmak için özeti Drive'a önbellekle
                try:
                    with open(summary_file, "w", encoding="utf-8") as sf:
                        json.dump({"wer": round(w, 2), "cer": round(c, 2), "samples": len(hyps)}, sf, indent=2)
                except Exception:
                    pass
                return float(w), float(c)
            except Exception:
                pass
    except Exception:
        pass
    return None, None


def main():
    args = parse_args()
    checkpoints_dir = os.path.abspath(os.path.expanduser(args.checkpoints_dir))

    print("=" * 80)
    print("  SLAM-ASR: GOOGLE DRIVE CHECKPOINT TEST ARACI (MASAÜSTÜ & YEREL)")
    print("=" * 80)
    print(f"[*] İşletim Sistemi:     {platform.system()} ({platform.release()})")
    print(f"[*] Google Drive Dizini: {checkpoints_dir}")
    print(f"[*] Test Manifestosu:    {args.cv_manifest}")

    if not os.path.exists(checkpoints_dir):
        print(f"\n[!] Belirtilen Google Drive checkpoints dizini bulunamadı:")
        print(f"    '{checkpoints_dir}'")
        user_input = input("\nLütfen Google Drive'ınızdaki checkpoints klasör yolunu girin: ").strip()
        if user_input and os.path.exists(os.path.abspath(os.path.expanduser(user_input))):
            checkpoints_dir = os.path.abspath(os.path.expanduser(user_input))
            print(f"[*] Kullanılacak Dizin: {checkpoints_dir}")
        else:
            print("[!] Geçerli bir dizin girilmedi. Çıkılıyor.")
            sys.exit(1)

    # 1. Manifest kontrolü
    cv_manifest = args.cv_manifest
    if not is_valid_manifest(cv_manifest):
        for cand in ["../cv/data/cv/test.jsonl", "../../cv/t0/data/cv/test.jsonl", "data/test.jsonl"]:
            if is_valid_manifest(cand):
                cv_manifest = cand
                print(f"[*] Alternatif Common Voice test manifestosu bulundu: {cv_manifest}")
                break

    if not is_valid_manifest(cv_manifest):
        print(f"\n[!] UYARI: Common Voice test ses dosyaları bulunamadı: {cv_manifest}")
        print("    Mevcut bir manifest yolu belirtmek için:")
        print("    python tr/tools/evaluate_local_checkpoints_on_cv.py --cv_manifest /dosya/yolu/data/cv/test.jsonl")
        print("\n    Veya veriyi indirmek için:")
        print("    python scripts/prepare_common_voice.py --download --api_key 'MDC_API_KEY' --output_dir data/cv")
        sys.exit(1)

    # 2. Checkpoint'leri listele
    checkpoints = find_checkpoints(checkpoints_dir, model_filter=args.models)
    if not checkpoints:
        print(f"\n[!] '{checkpoints_dir}' altında hiçbir 'best_checkpoint' bulunamadı.")
        print("    Lütfen Google Drive senkronizasyonunuzun tamamlandığından ve")
        print("    klasörde 'best_checkpoint' alt klasörleri bulunduğundan emin olun.")
        sys.exit(1)

    print(f"\n[*] Google Drive'da Tespit Edilen Model Sayısı: {len(checkpoints)}")
    print(f"{'No':<3} | {'Model Klasörü':<48} | {'CV Durumu'}")
    print("-" * 80)

    to_run = []
    for idx, (fname, fdir, ckpt) in enumerate(checkpoints, 1):
        eval_file = os.path.join(fdir, "eval_results_cv_generalization.jsonl")
        wer, cer = get_existing_metrics(eval_file)
        if wer is not None and not args.force:
            status_str = f"TAMAMLANDI (WER: {wer:.2f}%, CER: {cer:.2f}%)"
        else:
            status_str = "EKSİK (Test Edilecek)"
            to_run.append((idx, fname, fdir, ckpt, eval_file))
        print(f"{idx:<3} | {fname[:48]:<48} | {status_str}")

    print("-" * 80)

    if not to_run:
        print("\n[✓] Google Drive'daki tüm modellerin Common Voice değerlendirmesi zaten tamamlanmış!")
        print("    Yeniden çalıştırmak isterseniz '--force' bayrağını ekleyebilirsiniz.")
        return

    print(f"\n[*] Toplam {len(to_run)} eksik model sırayla test edilecek.")
    print("    (Sonuçlar doğrudan Google Drive'daki ilgili model klasörüne kaydedilip otomatik senkronize olacaktır)")
    confirm = input("\nDeğerlendirmeyi başlatmak istiyor musunuz? [E/h]: ").strip().lower()
    if confirm not in ("", "e", "evet", "y", "yes"):
        print("[*] İptal edildi.")
        return

    # Evaluate script
    eval_script = "tr/tsc/t5-ng-ta-gormez-tail-aug-lora/evaluate.py"
    if not os.path.exists(eval_script):
        eval_script = "evaluate.py"
    if not os.path.exists(eval_script):
        for cand in [
            "tr/tsc/t4-ng-ta-gormez-lora/evaluate.py",
            "tr/cv/t5-ng-ta-gormez-tail-aug-lora/evaluate.py",
            "../tr/tsc/t5-ng-ta-gormez-tail-aug-lora/evaluate.py",
        ]:
            if os.path.exists(cand):
                eval_script = cand
                break

    for idx, fname, fdir, ckpt, eval_file in to_run:
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(checkpoints)}] Test Başlatılıyor: {fname}")
        print("=" * 80)

        # Config dosyası: Önce Drive klasöründeki hardware_adaptive_turkish.yaml, yoksa repo içi fallback
        config_path = os.path.join(fdir, "hardware_adaptive_turkish.yaml")
        if not os.path.exists(config_path):
            config_path = "tr/tsc/t5-ng-ta-gormez-tail-aug-lora/config/hardware_adaptive_turkish.yaml"

        cmd = [
            sys.executable,
            eval_script,
            "--config", config_path,
            "--checkpoint", ckpt,
            "--manifest", cv_manifest,
            "--output_file", eval_file,
            "--beam_size", str(args.beam_size),
            "--language", "tr",
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        if args.batch_size:
            cmd.extend(["--batch_size", str(args.batch_size)])

        print(f"[*] Komut: {' '.join(cmd)}")
        ret = subprocess.run(cmd)

        if ret.returncode == 0 and os.path.exists(eval_file):
            w, c = get_existing_metrics(eval_file)
            w_str = f"{w:.2f}%" if w else "OK"
            c_str = f"{c:.2f}%" if c else "OK"
            print(f"\n[✓] BAŞARILI: {fname} -> WER: {w_str}, CER: {c_str}")
            print(f"    Sonuç Drive klasörüne işlendi: {eval_file}")
        else:
            print(f"\n[!] HATA: {fname} değerlendirilirken bir sorun oluştu.")

    print("\n" + "=" * 80)
    print("                    NİHAİ DEĞERLENDİRME TABLOSU")
    print("=" * 80)
    print(f"{'Model Adı':<48} | {'WER (%)':<10} | {'CER (%)':<10}")
    print("-" * 80)
    for _, (fname, fdir, _) in enumerate(checkpoints, 1):
        eval_file = os.path.join(fdir, "eval_results_cv_generalization.jsonl")
        w, c = get_existing_metrics(eval_file)
        w_str = f"{w:.2f}%" if w is not None else "Eksik"
        c_str = f"{c:.2f}%" if c is not None else "Eksik"
        print(f"{fname[:48]:<48} | {w_str:<10} | {c_str:<10}")
    print("=" * 80)


if __name__ == "__main__":
    main()
