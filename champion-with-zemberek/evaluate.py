#!/usr/bin/env python3
"""CLI entrypoint for evaluating SLAM-ASR models with Zemberek Language Model integration."""

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
from slam_asr.utils.zemberek_lm import ZemberekLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLAM-ASR Model with Zemberek LM")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to projector checkpoint (file or dir)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to test manifest (defaults to config test_manifest)")
    parser.add_argument("--output_file", type=str, default="eval_results.jsonl", help="Path to save predictions")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for generation (default: 4)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation (default: auto)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, mps, cpu)")
    parser.add_argument("--language", type=str, default="tr", help="Language for text normalization (tr or en)")
    # Zemberek LM arguments
    parser.add_argument("--use_zemberek_lm", action="store_true", default=True, help="Apply Zemberek Language Model support")
    parser.add_argument("--no_zemberek_lm", dest="use_zemberek_lm", action="store_false", help="Disable Zemberek Language Model")
    parser.add_argument("--lm_file", type=str, default=None, help="Path to precomputed Zemberek n-gram LM file")
    parser.add_argument("--train_manifest", type=str, default=None, help="Path to training manifest for in-domain LM estimation")
    parser.add_argument("--lm_weight", type=float, default=0.3, help="LM interpolation weight for rescoring (default: 0.3)")
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

    device_str = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device_str)

    # Hardware safety overrides
    if device.type != "cuda":
        print(f"[*] Notice: Running on {device.type}. Disabling 4-bit/8-bit quantization.")
        model_cfg_dict["load_in_4bit"] = False
        model_cfg_dict["load_in_8bit"] = False
        model_cfg_dict["torch_dtype"] = "float32"
    else:
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            if model_cfg_dict.get("torch_dtype") == "bfloat16":
                print("[*] Notice: bfloat16 not natively supported on this GPU. Using float16.")
                model_cfg_dict["torch_dtype"] = "float16"

    # Evaluation batch size safety (limit to 2 on <=16GB GPUs to avoid OOM during beam search)
    eval_batch_size = args.batch_size or data_cfg.get("batch_size", 2)
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb <= 16.5 and eval_batch_size > 2 and args.beam_size > 1:
            print(f"[*] Memory Optimization: Setting eval batch size to 2 on {vram_gb:.1f} GB GPU with beam_size={args.beam_size}.")
            eval_batch_size = 2

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
        weights_file = os.path.join(ckpt_path, "projector.pt")
        if not os.path.exists(weights_file):
            candidates = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith(".pt") or f.endswith(".bin")]
            if candidates:
                weights_file = candidates[0]
            else:
                raise FileNotFoundError(f"No checkpoint found in directory: {ckpt_path}")
    else:
        weights_file = ckpt_path

    print(f"[*] Loading checkpoint weights from: {weights_file}")
    checkpoint_data = torch.load(weights_file, map_location=device)
    if isinstance(checkpoint_data, dict) and "projector_state_dict" in checkpoint_data:
        raw_sd = checkpoint_data["projector_state_dict"]
    elif isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        raw_sd = checkpoint_data["model_state_dict"]
    elif isinstance(checkpoint_data, dict):
        raw_sd = checkpoint_data
    else:
        raise ValueError(f"Unexpected checkpoint format in {weights_file}")

    cleaned_sd = {}
    for k, v in raw_sd.items():
        new_k = k
        if new_k.startswith("projector."):
            new_k = new_k[len("projector."):]
        elif new_k.startswith("model.projector."):
            new_k = new_k[len("model.projector."):]
        cleaned_sd[new_k] = v

    try:
        model.projector.load_state_dict(cleaned_sd, strict=True)
    except Exception as load_err:
        print(f"[*] Strict state_dict load note: {load_err}. Retrying with strict=False...")
        model.projector.load_state_dict(cleaned_sd, strict=False)
    print("[*] Projector weights successfully loaded.")

    # Check for saved LoRA weights if configured
    if getattr(model_config, "use_lora", False):
        base_ckpt_dir = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
        candidates = [
            os.path.join(base_ckpt_dir, "lora_adapter"),
            base_ckpt_dir,
        ]
        lora_dir_found = None
        for candidate in candidates:
            if os.path.exists(candidate) and (
                os.path.exists(os.path.join(candidate, "adapter_config.json"))
                or os.path.exists(os.path.join(candidate, "adapter_model.safetensors"))
                or os.path.exists(os.path.join(candidate, "adapter_model.bin"))
            ):
                lora_dir_found = candidate
                break

        if lora_dir_found:
            print(f"[*] Loading LoRA adapter from: {lora_dir_found}")
            try:
                from peft import PeftModel
                if hasattr(model.language_model, "peft_config"):
                    model.language_model.load_adapter(lora_dir_found, adapter_name="default")
                else:
                    model.language_model = PeftModel.from_pretrained(model.language_model, lora_dir_found)
                print("[*] Successfully loaded LoRA adapter.")
            except Exception as e:
                print(f"[!] Warning: Could not load LoRA adapter ({e}), running with base language model.")

    model.eval()

    is_whisper = getattr(model_config, "encoder_type", "whisper") == "whisper"
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
    safe_workers = min(2, data_cfg.get("num_workers", 2), os.cpu_count() or 2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=safe_workers,
    )

    hypotheses = []
    lm_hypotheses = []
    references = []
    results = []

    # Initialize Zemberek Language Model if enabled
    zemberek_lm = None
    if args.use_zemberek_lm and args.language == "tr":
        train_man = args.train_manifest or data_cfg.get("train_manifest")
        manifest_to_use = train_man if (train_man and os.path.exists(train_man)) else None
        zemberek_lm = ZemberekLanguageModel(
            manifest_path=manifest_to_use,
            lm_file=args.lm_file,
        )
        print(f"[*] Zemberek Language Model initialized ({len(zemberek_lm.vocab):,} vocabulary terms, {len(zemberek_lm.bigrams):,} bigrams indexed).")

    model_dtype_str = getattr(model_config, "dtype", None) or getattr(model_config, "torch_dtype", "float32")
    use_amp = (model_dtype_str in ("float16", "bfloat16")) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (model_dtype_str == "bfloat16" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    autocast_ctx = (
        (lambda: torch.amp.autocast("cuda", dtype=amp_dtype))
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
        else (lambda: torch.cuda.amp.autocast(dtype=amp_dtype))
    )

    print(f"[*] Evaluating {len(test_dataset)} samples with beam_size={args.beam_size}...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Ensure speech inputs match encoder parameter dtype (avoid Float vs Half mismatch)
            enc_param = next(model.speech_encoder.parameters(), None)
            enc_dtype = enc_param.dtype if enc_param is not None else torch.float32

            speech_inputs = batch["speech_inputs"].to(device=device, dtype=enc_dtype)
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
        wer_gain = raw_wer - lm_wer
        cer_gain = raw_cer - lm_cer

        print("\n" + "=" * 65)
        print(f"[*] Evaluation Results on: {test_manifest}")
        print(f"[*] RAW ASR (Before Zemberek LM):")
        print(f"    • Word Error Rate (WER):      {raw_wer:.2f}%")
        print(f"    • Character Error Rate (CER): {raw_cer:.2f}%")
        print(f"[*] WITH ZEMBEREK LANGUAGE MODEL (Görmez, 2024 LM Support):")
        print(f"    • Word Error Rate (WER):      {lm_wer:.2f}%  (Net Kazanç: -{wer_gain:.2f}% mutlak)")
        print(f"    • Character Error Rate (CER): {lm_cer:.2f}%  (Net Kazanç: -{cer_gain:.2f}% mutlak)")
        print("=" * 65)
    else:
        print("\n" + "=" * 50)
        print(f"[*] Evaluation Results on: {test_manifest}")
        print(f"[*] Word Error Rate (WER):      {raw_wer:.2f}%")
        print(f"[*] Character Error Rate (CER): {raw_cer:.2f}%")
        print("=" * 50)

    # Print a few sample comparisons
    print("\n[*] Sample Predictions:")
    for sample in results[:5]:
        print(f"  Key: {sample['key']}")
        print(f"  REF:     {sample['norm_reference']}")
        print(f"  RAW HYP: {sample.get('norm_raw_hypothesis', sample['norm_hypothesis'])}")
        if zemberek_lm is not None:
            print(f"  LM  HYP: {sample['norm_lm_hypothesis']}")
        print("  " + "-" * 40)

    # Save to file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n[*] Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
