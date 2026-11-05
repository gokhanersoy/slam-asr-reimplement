#!/usr/bin/env python3
"""CLI entrypoint for evaluating SLAM-ASR models (t6-ng-ta-ctx-tail-aug-lora-lm) on test benchmarks."""

import argparse
import json
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from slam_asr.models.slam_model import SLAM_ASR, SLAMASRConfig
from slam_asr.data.audio_processor import AudioProcessor
from slam_asr.data.dataset import ASRDataset
from slam_asr.data.collator import ASRDataCollator
from slam_asr.utils.metrics import compute_wer, compute_cer, normalize_text
from slam_asr.utils.zemberek_lm import ZemberekLanguageModel
from slam_asr.utils.device_utils import resolve_auto_hyperparameters


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLAM-ASR Model (t6-ng-ta-ctx-tail-aug-lora-lm)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to projector checkpoint (file or dir)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to test manifest (defaults to config test_manifest)")
    parser.add_argument("--output_file", type=str, default="eval_results.jsonl", help="Path to save predictions")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for generation (default: 4)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation (default: auto)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, mps, cpu)")
    parser.add_argument("--language", type=str, default="tr", help="Language for text normalization (tr or en)")

    # Zemberek Language Model / Rescoring options
    parser.add_argument("--use_zemberek_lm", action="store_true", default=True, help="Apply Zemberek Language Model rescoring/morphology post-processing")
    parser.add_argument("--no_zemberek_lm", dest="use_zemberek_lm", action="store_false", help="Disable Zemberek Language Model rescoring")
    parser.add_argument("--zemberek_jar", type=str, default="tools/zemberek-full.jar", help="Path to zemberek-full.jar")
    parser.add_argument("--train_manifest", type=str, default=None, help="Path to training manifest for in-domain LM n-gram learning")
    parser.add_argument("--lexicon_file", type=str, default=None, help="Path to external lexicon file")
    parser.add_argument("--ngram_order", type=int, default=3, help="Max N-gram order for Zemberek LM (default: 3)")
    parser.add_argument("--lm_weight", type=float, default=0.35, help="LM interpolation weight (default: 0.35)")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_checkpoint(specified_path: str) -> str:
    """Resolve checkpoint file with fallback search if specified path does not exist."""
    import glob
    import re

    if specified_path and os.path.exists(specified_path):
        if os.path.isdir(specified_path):
            proj_file = os.path.join(specified_path, "projector.pt")
            if os.path.exists(proj_file):
                return proj_file
        return specified_path

    # Try common local alternatives
    local_candidates = [
        specified_path,
        os.path.join(specified_path, "projector.pt") if specified_path else "",
        "outputs/slam_asr_turkish_adaptive_t6_lm/best_checkpoint",
        "outputs/slam_asr_turkish_adaptive_t6_lm/best_checkpoint/projector.pt",
        "outputs/slam_asr_turkish_adaptive_t6_lm/final_checkpoint",
        "outputs/slam_asr_turkish_adaptive_t6_lm/final_checkpoint/projector.pt",
        "outputs/slam_asr_turkish_adaptive/best_checkpoint",
        "outputs/slam_asr_turkish_adaptive/best_checkpoint/projector.pt",
        "outputs/slam_asr_turkish_adaptive/final_checkpoint",
        "outputs/slam_asr_turkish_adaptive/final_checkpoint/projector.pt",
        "outputs/slam_asr_turkish_colab_t6_lm/best_checkpoint",
        "outputs/slam_asr_turkish_colab_t6_lm/best_checkpoint/projector.pt",
        "outputs/slam_asr_turkish_colab/best_checkpoint",
        "outputs/slam_asr_turkish_colab/best_checkpoint/projector.pt",
    ]
    for cand in local_candidates:
        if cand and os.path.exists(cand):
            target = os.path.join(cand, "projector.pt") if os.path.isdir(cand) else cand
            if os.path.exists(target):
                print(f"[*] Checkpoint '{specified_path}' not found, automatically using local: '{target}'")
                return target

    # Try Google Drive checkpoints (t6, t5, or other tsc checkpoints sorted by timestamp)
    drive_ckpts = sorted(
        glob.glob("/content/drive/MyDrive/checkpoints/*slam_asr_tr_tsc*/best_checkpoint") +
        glob.glob("/content/drive/MyDrive/checkpoints/*slam_asr_tr_tsc*/final_checkpoint"),
        key=lambda p: (re.search(r"(\d{8}_\d{6})", p).group(1) if re.search(r"(\d{8}_\d{6})", p) else "", p)
    )
    for d_cand in reversed(drive_ckpts):
        target = os.path.join(d_cand, "projector.pt") if os.path.isdir(d_cand) else d_cand
        if os.path.exists(target):
            print(f"[*] Checkpoint '{specified_path}' not found, automatically found in Google Drive: '{target}'")
            return target

    raise FileNotFoundError(
        f"Checkpoint not found at '{specified_path}'.\n"
        f"Lütfen önce 6. adımı çalıştırarak modeli eğitin veya Google Drive'da kayıtlı checkpoint yolunu kontrol edin."
    )


def resolve_manifest(specified_manifest: str, default_manifest: str) -> str:
    """Resolve test manifest path with fallback to Google Drive or sample validation data."""
    import glob
    import re

    path = specified_manifest or default_manifest
    if path and os.path.exists(path):
        return path

    # Check Google Drive backup manifests
    drive_manifests = sorted(
        glob.glob("/content/drive/MyDrive/checkpoints/*slam_asr_tr_tsc*/manifests/test.jsonl"),
        key=lambda p: (re.search(r"(\d{8}_\d{6})", p).group(1) if re.search(r"(\d{8}_\d{6})", p) else "", p)
    )
    if drive_manifests and os.path.exists(drive_manifests[-1]):
        print(f"[*] Test manifest not found at '{path}', automatically using Google Drive manifest: '{drive_manifests[-1]}'")
        return drive_manifests[-1]

    # Check local sample validation manifest
    sample_manifest = "data/sample_val_tr.jsonl"
    if os.path.exists(sample_manifest):
        print(f"[*] Test manifest not found at '{path}', fallback to sample validation data: '{sample_manifest}'")
        return sample_manifest

    raise FileNotFoundError(
        f"Test manifest not found at: '{path}'.\n"
        f"Lütfen 5. adımı çalıştırarak ISSAI TSC veri setini hazırlayın veya geçerli bir test manifestosu yolu belirtin."
    )


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    model_cfg_dict = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    lm_cfg = cfg.get("lm", {})

    # EARLY VALIDATION: Resolve checkpoint and test manifest BEFORE loading heavy models
    ckpt_path = find_checkpoint(args.checkpoint)
    test_manifest = resolve_manifest(args.manifest, data_cfg.get("test_manifest"))

    print(f"[*] Resolved Checkpoint: {ckpt_path}")
    print(f"[*] Resolved Test Manifest: {test_manifest}")

    # Resolve hardware-aware hyperparameters if auto
    model_cfg_dict, data_cfg, _ = resolve_auto_hyperparameters(
        model_cfg=model_cfg_dict,
        data_cfg=data_cfg,
        train_cfg={},
        cli_args=args,
        verbose=False,
    )

    eval_batch_size = args.batch_size or data_cfg.get("batch_size", 4)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device_str)

    print(f"[*] Loading model on {device}...")
    model_config = SLAMASRConfig(**model_cfg_dict)
    model = SLAM_ASR(config=model_config)
    if not getattr(model_config, "load_in_4bit", False) and not getattr(model_config, "load_in_8bit", False):
        model = model.to(device)
    else:
        model.speech_encoder = model.speech_encoder.to(device)
        model.projector = model.projector.to(device)

    # Load projector weights
    print(f"[*] Loading projector weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.projector.load_state_dict(state_dict)

    # Check for trained LoRA adapter in checkpoint directory
    ckpt_dir = os.path.dirname(ckpt_path) if ckpt_path.endswith(".pt") else ckpt_path
    lora_dir = os.path.join(ckpt_dir, "lora_adapter")
    if os.path.exists(lora_dir) and os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        try:
            from peft import PeftModel
            print(f"[*] Loading trained LoRA adapter from: {lora_dir}")
            if hasattr(model.language_model, "peft_config"):
                model.language_model.load_adapter(lora_dir, adapter_name="default")
            else:
                model.language_model = PeftModel.from_pretrained(model.language_model, lora_dir)
            print("[*] Successfully loaded LoRA adapter.")
        except Exception as e:
            print(f"[!] Warning: Could not load LoRA adapter ({e}), running with base language model.")

    model.eval()

    # Data loading
    is_whisper = data_cfg.get("is_whisper", False)
    audio_processor = AudioProcessor(
        sampling_rate=data_cfg.get("sampling_rate", 16000),
        normalize=data_cfg.get("normalize", False),
        is_whisper=is_whisper,
        whisper_mel_bins=data_cfg.get("whisper_mel_bins", 128 if is_whisper else 80),
    )

    test_manifest = args.manifest or data_cfg.get("test_manifest")
    if not test_manifest or not os.path.exists(test_manifest):
        raise FileNotFoundError(f"Test manifest not found at: {test_manifest}")

    test_dataset = ASRDataset(
        data_source=test_manifest,
        audio_processor=audio_processor,
        prompt=model_config.default_prompt,
    )
    collator = ASRDataCollator(is_whisper=is_whisper)
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=data_cfg.get("num_workers", 2),
    )

    hypotheses = []
    lm_hypotheses = []
    references = []
    results = []

    # Initialize Zemberek Language Model if requested
    zemberek_lm = None
    if args.use_zemberek_lm and args.language == "tr":
        manifest_path = args.train_manifest or data_cfg.get("train_manifest")
        manifest_to_use = manifest_path if (manifest_path and os.path.exists(manifest_path)) else None
        jar_path = args.zemberek_jar or lm_cfg.get("zemberek_jar_path")
        ngram_order = args.ngram_order or lm_cfg.get("ngram_order", 3)
        lm_weight = args.lm_weight or lm_cfg.get("lm_weight", 0.35)

        zemberek_lm = ZemberekLanguageModel(
            zemberek_jar_path=jar_path if (jar_path and os.path.exists(jar_path)) else None,
            train_manifest=manifest_to_use,
            lexicon_file=args.lexicon_file or lm_cfg.get("lexicon_file"),
            ngram_order=ngram_order,
            lm_weight=lm_weight,
        )
        print(f"[*] Zemberek Language Model initialized ({len(zemberek_lm.vocab):,} vocabulary tokens indexed).")

    model_dtype_str = getattr(model_config, "dtype", None) or getattr(model_config, "torch_dtype", "float32")
    use_amp = (model_dtype_str in ("float16", "bfloat16")) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if model_dtype_str == "bfloat16" else torch.float16
    autocast_ctx = (
        (lambda: torch.amp.autocast("cuda", dtype=amp_dtype))
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
        else (lambda: torch.cuda.amp.autocast(dtype=amp_dtype))
    )

    print(f"[*] Evaluating {len(test_dataset)} samples with beam_size={args.beam_size}...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            speech_inputs = batch["speech_inputs"].to(device)
            speech_mask = (
                batch["speech_attention_mask"].to(device)
                if batch["speech_attention_mask"] is not None
                else None
            )

            if use_amp:
                with autocast_ctx():
                    preds = model.generate(
                        speech_inputs=speech_inputs,
                        speech_attention_mask=speech_mask,
                        prompts=batch["prompts"],
                        num_beams=args.beam_size,
                    )
            else:
                preds = model.generate(
                    speech_inputs=speech_inputs,
                    speech_attention_mask=speech_mask,
                    prompts=batch["prompts"],
                    num_beams=args.beam_size,
                )

            for key, pred, ref in zip(batch["keys"], preds, batch["transcripts"]):
                lm_pred = zemberek_lm.correct_sentence(pred) if zemberek_lm is not None else pred
                hypotheses.append(pred)
                lm_hypotheses.append(lm_pred)
                references.append(ref)

                results.append({
                    "key": key,
                    "reference": ref,
                    "hypothesis": lm_pred if zemberek_lm is not None else pred,
                    "raw_hypothesis": pred,
                    "lm_hypothesis": lm_pred,
                    "norm_reference": normalize_text(ref, language=args.language),
                    "norm_hypothesis": normalize_text(lm_pred if zemberek_lm is not None else pred, language=args.language),
                    "norm_raw_hypothesis": normalize_text(pred, language=args.language),
                    "norm_lm_hypothesis": normalize_text(lm_pred, language=args.language),
                })

    raw_wer = compute_wer(hypotheses, references, language=args.language)
    raw_cer = compute_cer(hypotheses, references, language=args.language)

    if zemberek_lm is not None:
        lm_wer = compute_wer(lm_hypotheses, references, language=args.language)
        lm_cer = compute_cer(lm_hypotheses, references, language=args.language)
        wer_diff = lm_wer - raw_wer
        cer_diff = lm_cer - raw_cer

        print("\n" + "=" * 70)
        print(f"[*] Evaluation Results on: {test_manifest}")
        print(f"[*] RAW ASR (End-to-End Multimodal Model - t5):")
        print(f"    • Word Error Rate (WER):      {raw_wer:.2f}%")
        print(f"    • Character Error Rate (CER): {raw_cer:.2f}%")
        print(f"[*] WITH ZEMBEREK LANGUAGE MODEL RESCORING (t6-lm):")
        print(f"    • Word Error Rate (WER):      {lm_wer:.2f}%  (Fark: {wer_diff:+.2f}%)")
        print(f"    • Character Error Rate (CER): {lm_cer:.2f}%  (Fark: {cer_diff:+.2f}%)")
        print("=" * 70)

        # Print sample comparisons showing Zemberek LM impact
        print("\n[*] Sample Predictions (Zemberek LM Impact):")
        for sample in results[:5]:
            print(f"  Key: {sample['key']}")
            print(f"  REF:     {sample['norm_reference']}")
            print(f"  RAW HYP: {sample['norm_raw_hypothesis']}")
            print(f"  LM HYP:  {sample['norm_lm_hypothesis']}")
            print("  " + "-" * 50)
    else:
        print("\n" + "=" * 50)
        print(f"[*] Evaluation Results on: {test_manifest}")
        print(f"[*] Word Error Rate (WER):     {raw_wer:.2f}%")
        print(f"[*] Character Error Rate (CER): {raw_cer:.2f}%")
        print("=" * 50)

        # Print a few sample comparisons
        print("\n[*] Sample Predictions:")
        for sample in results[:5]:
            print(f"  Key: {sample['key']}")
            print(f"  REF: {sample['norm_reference']}")
            print(f"  HYP: {sample['norm_hypothesis']}")
            print("  " + "-" * 40)

    # Save to file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n[*] Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
