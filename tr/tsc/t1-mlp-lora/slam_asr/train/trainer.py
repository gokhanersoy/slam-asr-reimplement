"""Training loop and Trainer class for SLAM-ASR."""

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from slam_asr.models.slam_model import SLAM_ASR
from slam_asr.utils.scheduler import get_warmup_constant_scheduler, get_warmup_cosine_scheduler


@dataclass
class TrainingArguments:
    """Hyperparameters and runtime settings for SLAM-ASR training."""

    output_dir: str = "./outputs"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    max_steps: int = 100000
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: int = 2000
    early_stopping_patience: int = 5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class ASRTrainer:
    """Trainer tailored for SLAM-ASR adapter training."""

    def __init__(
        self,
        model: SLAM_ASR,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        args: Optional[TrainingArguments] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        self.args = args or TrainingArguments()
        self.device = torch.device(self.args.device)
        if getattr(model.config, "load_in_4bit", False) or getattr(model.config, "load_in_8bit", False):
            model.speech_encoder = model.speech_encoder.to(self.device)
            model.projector = model.projector.to(self.device)
            self.model = model
        else:
            self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        os.makedirs(self.args.output_dir, exist_ok=True)

        # Get trainable parameters (only the projector by default)
        trainable_params = self.model.get_trainable_parameters()
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found! Ensure projector is not frozen.")

        # Optimizer: AdamW without weight decay (Paper Section 3.3)
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # Scheduler: Warmup 1000 steps then constant max learning rate
        self.lr_scheduler = lr_scheduler or get_warmup_constant_scheduler(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
        )

        # Mixed precision scaler
        self.use_amp = (self.args.fp16 or self.args.bf16) and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.args.fp16 and self.device.type == "cuda")
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16 and self.device.type == "cuda")
        self.amp_dtype = amp_dtype

    def _autocast(self):
        """Unified autocast context manager supporting modern PyTorch 2.x and legacy."""
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        return torch.cuda.amp.autocast(dtype=self.amp_dtype)

    def train(self):
        """Execute full training loop with early stopping."""
        print(f"[*] Starting SLAM-ASR Training on device: {self.device}")
        trainable_count = sum(p.numel() for p in self.model.get_trainable_parameters())
        total_count = sum(p.numel() for p in self.model.parameters())
        print(f"[*] Trainable parameters: {trainable_count:,} / {total_count:,} ({100.0 * trainable_count / total_count:.2f}%)")

        global_step = 0
        best_eval_loss = float("inf")
        patience_counter = 0
        running_loss = 0.0

        self.model.train()
        epoch = 0

        progress_bar = tqdm(total=self.args.max_steps, desc="Training")

        while global_step < self.args.max_steps:
            epoch += 1
            for step, batch in enumerate(self.train_dataloader):
                speech_inputs = batch["speech_inputs"].to(self.device)
                speech_attention_mask = (
                    batch["speech_attention_mask"].to(self.device)
                    if batch["speech_attention_mask"] is not None
                    else None
                )
                transcripts = batch["transcripts"]
                prompts = batch["prompts"]

                # Mixed precision forward pass
                if self.use_amp:
                    with self._autocast():
                        outputs = self.model(
                            speech_inputs=speech_inputs,
                            speech_attention_mask=speech_attention_mask,
                            transcripts=transcripts,
                            prompts=prompts,
                        )
                        loss = outputs["loss"] / self.args.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(
                        speech_inputs=speech_inputs,
                        speech_attention_mask=speech_attention_mask,
                        transcripts=transcripts,
                        prompts=prompts,
                    )
                    loss = outputs["loss"] / self.args.gradient_accumulation_steps
                    loss.backward()

                running_loss += loss.item() * self.args.gradient_accumulation_steps

                # Optimizer step after accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.get_trainable_parameters(),
                            self.args.max_grad_norm,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.get_trainable_parameters(),
                            self.args.max_grad_norm,
                        )
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    # Logging
                    if global_step % self.args.logging_steps == 0:
                        avg_loss = running_loss / self.args.logging_steps
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
                        running_loss = 0.0

                    # Evaluation
                    if self.eval_dataloader is not None and global_step % self.args.eval_steps == 0:
                        eval_loss = self.evaluate()
                        print(f"\n[Step {global_step}] Validation Loss: {eval_loss:.4f}")

                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            patience_counter = 0
                            self.save_checkpoint(f"best_checkpoint")
                            print(f"[*] New best validation loss achieved: {best_eval_loss:.4f}")
                        else:
                            patience_counter += 1
                            if patience_counter >= self.args.early_stopping_patience:
                                print(f"[*] Early stopping triggered after {patience_counter} evals without improvement.")
                                self.save_checkpoint("final_checkpoint")
                                progress_bar.close()
                                return

                        self.model.train()

                    # Periodic Save
                    if global_step % self.args.save_steps == 0:
                        self.save_checkpoint(f"step_{global_step}")

                    if global_step >= self.args.max_steps:
                        break

        progress_bar.close()
        self.save_checkpoint("final_checkpoint")
        print("[*] Training completed.")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute average validation loss on evaluation dataloader."""
        if self.eval_dataloader is None:
            return 0.0

        self.model.eval()
        total_eval_loss = 0.0
        total_batches = 0

        for batch in self.eval_dataloader:
            speech_inputs = batch["speech_inputs"].to(self.device)
            speech_attention_mask = (
                batch["speech_attention_mask"].to(self.device)
                if batch["speech_attention_mask"] is not None
                else None
            )
            transcripts = batch["transcripts"]
            prompts = batch["prompts"]

            if self.use_amp:
                with self._autocast():
                    outputs = self.model(
                        speech_inputs=speech_inputs,
                        speech_attention_mask=speech_attention_mask,
                        transcripts=transcripts,
                        prompts=prompts,
                    )
            else:
                outputs = self.model(
                    speech_inputs=speech_inputs,
                    speech_attention_mask=speech_attention_mask,
                    transcripts=transcripts,
                    prompts=prompts,
                )

            total_eval_loss += outputs["loss"].item()
            total_batches += 1

        return total_eval_loss / max(1, total_batches)

    def save_checkpoint(self, tag: str):
        """Save projector weights, LoRA adapter, and metadata."""
        ckpt_dir = os.path.join(self.args.output_dir, tag)
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. Save projector weights
        projector_path = os.path.join(ckpt_dir, "projector.pt")
        torch.save(self.model.projector.state_dict(), projector_path)

        # 2. Save LoRA adapter ONLY if PEFT model is active (never save the base 15GB LLM!)
        if hasattr(self.model.language_model, "peft_config"):
            lora_dir = os.path.join(ckpt_dir, "lora_adapter")
            self.model.language_model.save_pretrained(lora_dir, safe_serialization=True)
            print(f"[*] Saved LoRA adapter to {lora_dir}")

        # 3. Save config dict
        config_path = os.path.join(ckpt_dir, "config.pt")
        torch.save(self.model.config, config_path)

        print(f"[*] Saved checkpoint to {ckpt_dir}")
