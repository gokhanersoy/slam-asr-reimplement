#!/usr/bin/env python3
"""CLI entrypoint for training SLAM-ASR models."""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from slam_asr.models.slam_model import SLAM_ASR, SLAMASRConfig
from slam_asr.data.audio_processor import AudioProcessor
from slam_asr.data.dataset import ASRDataset
from slam_asr.data.collator import ASRDataCollator
from slam_asr.train.trainer import ASRTrainer, TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser(description="Train SLAM-ASR Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda, mps, cpu)")
    parser.add_argument("--train_manifest", type=str, default=None, help="Override train manifest path")
    parser.add_argument("--val_manifest", type=str, default=None, help="Override validation manifest path")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cli_args = parse_args()
    cfg = load_yaml_config(cli_args.config)

    model_cfg_dict = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg_dict = cfg.get("training", {})

    # Overrides
    if cli_args.output_dir:
        train_cfg_dict["output_dir"] = cli_args.output_dir
    if cli_args.learning_rate:
        train_cfg_dict["learning_rate"] = cli_args.learning_rate
    if cli_args.max_steps:
        train_cfg_dict["max_steps"] = cli_args.max_steps
    if cli_args.device:
        train_cfg_dict["device"] = cli_args.device

    batch_size = cli_args.batch_size or data_cfg.get("batch_size", 6)

    print(f"[*] Initializing SLAM-ASR with config: {cli_args.config}")
    model_config = SLAMASRConfig(**model_cfg_dict)
    model = SLAM_ASR(config=model_config)

    # Audio Processor
    is_whisper = data_cfg.get("is_whisper", False)
    whisper_mel_bins = data_cfg.get("whisper_mel_bins", 80)
    audio_processor = AudioProcessor(
        sampling_rate=data_cfg.get("sampling_rate", 16000),
        normalize=data_cfg.get("normalize", False),
        is_whisper=is_whisper,
        whisper_mel_bins=whisper_mel_bins,
    )

    # Datasets
    train_manifest = cli_args.train_manifest or data_cfg.get("train_manifest")
    if not train_manifest or not os.path.exists(train_manifest):
        if train_manifest == "data/colab_train.jsonl" and os.path.exists("data/sample_train.jsonl"):
            print(
                f"[*] Notice: '{train_manifest}' not found yet. Falling back to toy dataset 'data/sample_train.jsonl'.\n"
                f"    (If you intend to train on LibriSpeech, please run Step 5 first to download the data!)"
            )
            train_manifest = "data/sample_train.jsonl"
        else:
            raise FileNotFoundError(
                f"Train manifest not found at: '{train_manifest}'.\n"
                f"  - If running toy test (Step 4), run: python scripts/create_toy_dataset.py\n"
                f"  - If running real training (Step 6), make sure Step 5 (LibriSpeech prep) finished."
            )

    train_dataset = ASRDataset(
        data_source=train_manifest,
        audio_processor=audio_processor,
        prompt=model_config.default_prompt,
    )

    val_dataset = None
    val_manifest = cli_args.val_manifest or data_cfg.get("val_manifest")
    if val_manifest and not os.path.exists(val_manifest) and os.path.exists("data/sample_val.jsonl"):
        val_manifest = "data/sample_val.jsonl"

    if val_manifest and os.path.exists(val_manifest):
        val_dataset = ASRDataset(
            data_source=val_manifest,
            audio_processor=audio_processor,
            prompt=model_config.default_prompt,
        )

    # Collator & DataLoaders
    collator = ASRDataCollator(is_whisper=is_whisper)
    num_workers = data_cfg.get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        )

    # Trainer
    train_args = TrainingArguments(**train_cfg_dict)
    trainer = ASRTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        args=train_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
