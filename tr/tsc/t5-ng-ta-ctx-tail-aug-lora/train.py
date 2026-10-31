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
from slam_asr.utils.device_utils import resolve_auto_hyperparameters


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
    parser.add_argument("--auto_device", action="store_true", default=False, help="Auto-optimize hyperparameters for active GPU/hardware")
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

    # Automatically resolve hardware-aware hyperparameters based on detected GPU / VRAM
    model_cfg_dict, data_cfg, train_cfg_dict = resolve_auto_hyperparameters(
        model_cfg=model_cfg_dict,
        data_cfg=data_cfg,
        train_cfg=train_cfg_dict,
        cli_args=cli_args,
        verbose=True,
    )

    # CLI Overrides (explicit user arguments always take precedence)
    if cli_args.output_dir:
        train_cfg_dict["output_dir"] = cli_args.output_dir
    if cli_args.learning_rate:
        train_cfg_dict["learning_rate"] = cli_args.learning_rate
    if cli_args.max_steps:
        train_cfg_dict["max_steps"] = cli_args.max_steps
    if cli_args.device:
        train_cfg_dict["device"] = cli_args.device

    batch_size = cli_args.batch_size or data_cfg.get("batch_size", 4)

    print(f"[*] Initializing SLAM-ASR with config: {cli_args.config}")
    model_config = SLAMASRConfig(**model_cfg_dict)
    model = SLAM_ASR(config=model_config)

    # Audio Processors (Targeted Tail Augmentation for Training, Clean for Evaluation)
    is_whisper = data_cfg.get("is_whisper", False)
    whisper_mel_bins = data_cfg.get("whisper_mel_bins", 80)
    train_audio_processor = AudioProcessor(
        sampling_rate=data_cfg.get("sampling_rate", 16000),
        normalize=data_cfg.get("normalize", False),
        is_whisper=is_whisper,
        whisper_mel_bins=whisper_mel_bins,
        apply_tail_speedup=data_cfg.get("apply_tail_speedup", True),
        tail_speedup_range=tuple(data_cfg.get("tail_speedup_range", [1.15, 1.35])),
        apply_tail_energy_decay=data_cfg.get("apply_tail_energy_decay", True),
        tail_decay_db_range=tuple(data_cfg.get("tail_decay_db_range", [-3.0, -8.0])),
        apply_tail_spec_masking=data_cfg.get("apply_tail_spec_masking", True),
        tail_ratio_range=tuple(data_cfg.get("tail_ratio_range", [0.20, 0.35])),
        tail_aug_prob=data_cfg.get("tail_aug_prob", 0.8),
        is_training=True,
    )
    eval_audio_processor = AudioProcessor(
        sampling_rate=data_cfg.get("sampling_rate", 16000),
        normalize=data_cfg.get("normalize", False),
        is_whisper=is_whisper,
        whisper_mel_bins=whisper_mel_bins,
        is_training=False,
    )

    # Datasets
    train_manifest = cli_args.train_manifest or data_cfg.get("train_manifest")
    if not train_manifest or not os.path.exists(train_manifest):
        if os.path.exists("data/sample_train_tr.jsonl"):
            print(
                f"[*] Notice: '{train_manifest}' not found yet. Falling back to Turkish toy dataset 'data/sample_train_tr.jsonl'.\n"
                f"    (To train on full ISSAI TSC data, please run 'python scripts/prepare_issai_tsc.py --download' first!)"
            )
            train_manifest = "data/sample_train_tr.jsonl"
        elif os.path.exists("data/sample_train.jsonl"):
            train_manifest = "data/sample_train.jsonl"
        else:
            raise FileNotFoundError(
                f"Train manifest not found at: '{train_manifest}'.\n"
                f"  - For quick testing, run: python scripts/create_toy_turkish_dataset.py\n"
                f"  - For real training, run: python scripts/prepare_issai_tsc.py --download"
            )

    train_dataset = ASRDataset(
        data_source=train_manifest,
        audio_processor=train_audio_processor,
        prompt=model_config.default_prompt,
    )

    val_dataset = None
    val_manifest = cli_args.val_manifest or data_cfg.get("val_manifest")
    if val_manifest and not os.path.exists(val_manifest):
        if os.path.exists("data/sample_val_tr.jsonl"):
            val_manifest = "data/sample_val_tr.jsonl"
        elif os.path.exists("data/sample_val.jsonl"):
            val_manifest = "data/sample_val.jsonl"

    if val_manifest and os.path.exists(val_manifest):
        val_dataset = ASRDataset(
            data_source=val_manifest,
            audio_processor=eval_audio_processor,
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
