#!/usr/bin/env python3
"""CLI entrypoint for evaluating SLAM-ASR models on test benchmarks."""

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
from slam_asr.utils.device_utils import resolve_auto_hyperparameters


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLAM-ASR Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to projector checkpoint (file or dir)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to test manifest (defaults to config test_manifest)")
    parser.add_argument("--output_file", type=str, default="eval_results.jsonl", help="Path to save predictions")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for generation (default: 4)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation (default: auto)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, mps, cpu)")
    parser.add_argument("--language", type=str, default="tr", help="Language for text normalization (tr or en)")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    model_cfg_dict = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

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

    # Load projector checkpoint
    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "projector.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Projector checkpoint not found at: {ckpt_path}")

    print(f"[*] Loading projector weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.projector.load_state_dict(state_dict)
    model.eval()

    # Data loading
    is_whisper = data_cfg.get("is_whisper", False)
    audio_processor = AudioProcessor(
        sampling_rate=data_cfg.get("sampling_rate", 16000),
        normalize=data_cfg.get("normalize", False),
        is_whisper=is_whisper,
        whisper_mel_bins=data_cfg.get("whisper_mel_bins", 80),
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
    references = []
    results = []

    use_amp = (model_config.torch_dtype in ("float16", "bfloat16")) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float16
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
                hypotheses.append(pred)
                references.append(ref)
                results.append({
                    "key": key,
                    "reference": ref,
                    "hypothesis": pred,
                    "norm_reference": normalize_text(ref, language=args.language),
                    "norm_hypothesis": normalize_text(pred, language=args.language),
                })

    wer = compute_wer(hypotheses, references, language=args.language)
    cer = compute_cer(hypotheses, references, language=args.language)

    print("\n" + "=" * 50)
    print(f"[*] Evaluation Results on: {test_manifest}")
    print(f"[*] Word Error Rate (WER):     {wer:.2f}%")
    print(f"[*] Character Error Rate (CER): {cer:.2f}%")
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
            f.write(json.dumps(item) + "\n")
    print(f"\n[*] Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
