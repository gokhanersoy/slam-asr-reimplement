#!/usr/bin/env python3
"""CLI script to transcribe audio files using SLAM-ASR with Zemberek Language Model."""

import argparse
import os
import yaml
import torch

from slam_asr.models.slam_model import SLAM_ASR, SLAMASRConfig
from slam_asr.data.audio_processor import AudioProcessor
from slam_asr.utils.device_utils import resolve_auto_hyperparameters
from slam_asr.utils.zemberek_lm import ZemberekLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio using SLAM-ASR with Zemberek LM")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav, .flac, .mp3)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to projector weights (projector.pt or dir)")
    parser.add_argument("--prompt", type=str, default="Konuşmayı metne dönüştürün.", help="Instruction prompt")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for decoding")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--use_zemberek_lm", action="store_true", default=True, help="Apply Zemberek Language Model correction")
    parser.add_argument("--no_zemberek_lm", dest="use_zemberek_lm", action="store_false", help="Disable Zemberek LM")
    parser.add_argument("--lm_file", type=str, default=None, help="Path to precomputed Zemberek n-gram LM file")
    parser.add_argument("--train_manifest", type=str, default=None, help="Path to training manifest for in-domain LM")
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

    # Check for trained LoRA adapter in checkpoint directory
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

    # Process audio
    is_whisper = getattr(model_config, "encoder_type", "whisper") == "whisper"
    audio_processor = AudioProcessor(
        sampling_rate=data_cfg.get("sampling_rate", 16000),
        normalize=data_cfg.get("normalize", False),
        is_whisper=is_whisper,
        whisper_mel_bins=data_cfg.get("whisper_mel_bins", 80),
    )

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    print(f"[*] Processing audio: {args.audio}")
    enc_param = next(model.speech_encoder.parameters(), None)
    enc_dtype = enc_param.dtype if enc_param is not None else torch.float32
    speech_tensor = audio_processor.process(args.audio).unsqueeze(0).to(device=device, dtype=enc_dtype)

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
                raw_transcription = model.generate(
                    speech_inputs=speech_tensor,
                    prompts=[args.prompt],
                    num_beams=args.beam_size,
                )[0]
        else:
            raw_transcription = model.generate(
                speech_inputs=speech_tensor,
                prompts=[args.prompt],
                num_beams=args.beam_size,
            )[0]

    lm_transcription = raw_transcription
    ppl = None
    log_prob = None
    if args.use_zemberek_lm:
        train_man = args.train_manifest or data_cfg.get("train_manifest")
        manifest_to_use = train_man if (train_man and os.path.exists(train_man)) else None
        zemberek_lm = ZemberekLanguageModel(
            manifest_path=manifest_to_use,
            lm_file=args.lm_file,
        )
        lm_transcription = zemberek_lm.correct_sentence(raw_transcription)
        log_prob = zemberek_lm.score_sentence(lm_transcription)
        ppl = zemberek_lm.perplexity(lm_transcription)

    print("\n" + "=" * 65)
    print(f"[*] Audio File:            {args.audio}")
    print(f"[*] Prompt:                {args.prompt}")
    print(f"[*] Raw ASR (Champion t5): {raw_transcription}")
    if args.use_zemberek_lm:
        print(f"[*] With Zemberek LM:      {lm_transcription}")
        if ppl is not None:
            print(f"[*] LM Perplexity:         {ppl:.2f}")
            print(f"[*] LM Log-Likelihood:     {log_prob:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
