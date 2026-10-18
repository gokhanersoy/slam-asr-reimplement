#!/usr/bin/env python3
"""CLI script to transcribe audio files using SLAM-ASR."""

import argparse
import os
import yaml
import torch

from slam_asr.models.slam_model import SLAM_ASR, SLAMASRConfig
from slam_asr.data.audio_processor import AudioProcessor
from slam_asr.utils.device_utils import resolve_auto_hyperparameters


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio using SLAM-ASR")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav, .flac, .mp3)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to projector weights (projector.pt)")
    parser.add_argument("--prompt", type=str, default="Konuşmayı metne dönüştürün.", help="Instruction prompt")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for decoding")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
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

    device_str = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device_str)

    print(f"[*] Loading SLAM-ASR on {device}...")
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
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"[*] Loading projector weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.projector.load_state_dict(state_dict)

    # Check for trained LoRA adapter in checkpoint directory
    ckpt_dir = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
    lora_dir = os.path.join(ckpt_dir, "lora_adapter")
    if os.path.exists(lora_dir):
        try:
            from peft import PeftModel
            print(f"[*] Loading trained LoRA adapter from: {lora_dir}")
            model.language_model = PeftModel.from_pretrained(model.language_model, lora_dir)
        except ImportError:
            print("[!] Warning: peft not installed, running with base language model.")

    model.eval()

    # Process audio
    is_whisper = data_cfg.get("is_whisper", False)
    audio_processor = AudioProcessor(
        sampling_rate=data_cfg.get("sampling_rate", 16000),
        normalize=data_cfg.get("normalize", False),
        is_whisper=is_whisper,
        whisper_mel_bins=data_cfg.get("whisper_mel_bins", 80),
    )

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    print(f"[*] Processing audio: {args.audio}")
    speech_tensor = audio_processor.process(args.audio).unsqueeze(0).to(device)  # (1, samples) or (1, mels, 3000)

    model_dtype_str = getattr(model_config, "dtype", None) or getattr(model_config, "torch_dtype", "float32")
    use_amp = (model_dtype_str in ("float16", "bfloat16")) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if model_dtype_str == "bfloat16" else torch.float16
    autocast_ctx = (
        (lambda: torch.amp.autocast("cuda", dtype=amp_dtype))
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
        else (lambda: torch.cuda.amp.autocast(dtype=amp_dtype))
    )

    print(f"[*] Transcribing (beam_size={args.beam_size})...")
    with torch.no_grad():
        if use_amp:
            with autocast_ctx():
                transcription = model.generate(
                    speech_inputs=speech_tensor,
                    prompts=[args.prompt],
                    num_beams=args.beam_size,
                )[0]
        else:
            transcription = model.generate(
                speech_inputs=speech_tensor,
                prompts=[args.prompt],
                num_beams=args.beam_size,
            )[0]

    print("\n" + "=" * 50)
    print(f"Audio:          {args.audio}")
    print(f"Prompt:         {args.prompt}")
    print(f"Transcription:  {transcription}")
    print("=" * 50)


if __name__ == "__main__":
    main()
