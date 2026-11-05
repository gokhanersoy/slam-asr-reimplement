#!/usr/bin/env python3
"""CLI entrypoint for evaluating SLAM-ASR t6-ng-ta-ctx-tail-aug-lora-lm on test benchmarks."""

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


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    model_cfg_dict = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    lm_cfg = cfg.get("lm", {})

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

    # Check for trained LoRA adapter in checkpoint directory
    ckpt_dir = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
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
        is_whisper=is_whisper,
        whisper_mel_bins=data_cfg.get("whisper_mel_bins", 128),
    )

    test_manifest = args.manifest or data_cfg.get("test_manifest")
    if not test_manifest or not os.path.exists(test_manifest):
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}")

    test_dataset = ASRDataset(
        manifest_path=test_manifest,
        audio_processor=audio_processor,
        tokenizer=model.tokenizer,
        default_prompt=model_config.default_prompt,
        user_prefix=model_config.user_prefix,
        assistant_prefix=model_config.assistant_prefix,
    )

    collator = ASRDataCollator(
        pad_token_id=model.tokenizer.pad_token_id or 0,
        is_whisper=is_whisper,
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=collator,
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

    print(f"[*] Evaluating on: {test_manifest} ({len(test_dataset)} utterances)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            speech_values = batch["speech_values"].to(device)
            speech_lengths = batch["speech_lengths"].to(device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    preds = model.generate(
                        speech_values=speech_values,
                        speech_lengths=speech_lengths,
                        prompts=batch["prompts"],
                        beam_size=args.beam_size,
                    )
            else:
                preds = model.generate(
                    speech_values=speech_values,
                    speech_lengths=speech_lengths,
                    prompts=batch["prompts"],
                    beam_size=args.beam_size,
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
