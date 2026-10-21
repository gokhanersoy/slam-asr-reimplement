"""Targeted Speech Tail Augmentations for SLAM-ASR (t1-tail-aug).

Addresses the "Utterance-Final Phoneme Swallowing / Energy Drop" phenomenon:
In Turkish (an agglutinative, suffix-heavy language), speakers frequently accelerate,
drop vocal intensity, and reduce vowels/consonants in the final 20-30% of a sentence
as respiratory pressure diminishes before pauses.

This module provides:
1. TailSpeedPerturbation: Selectively accelerates the final 20-35% of an utterance (1.15x - 1.35x)
   with smooth cross-fade at the boundary, training the speech encoder and projector
   to reconstruct words from temporally compressed acoustic frames.
2. TailEnergyDecay: Simulates respiratory energy drop by linearly fading out the tail by -3 dB to -8 dB.
3. TailSpecMasking: Concentrates time-frequency masking on the speech tail region in log-Mel space.
4. TailAugmentationPipeline: Unified online augmentation pipeline for training.
"""

import random
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class TailSpeedPerturbation:
    """Selectively accelerates the final section of an utterance.

    Args:
        tail_ratio_range: (min_ratio, max_ratio) fraction of audio treated as tail (default: (0.20, 0.35)).
        speed_factor_range: (min_speed, max_speed) acceleration factor (default: (1.15, 1.35)).
        crossfade_ms: Duration of linear cross-fade window in milliseconds to prevent clicks (default: 40).
        sample_rate: Expected audio sampling rate in Hz (default: 16000).
        p: Probability of applying augmentation (default: 0.8).
    """

    def __init__(
        self,
        tail_ratio_range: Tuple[float, float] = (0.20, 0.35),
        speed_factor_range: Tuple[float, float] = (1.15, 1.35),
        crossfade_ms: int = 40,
        sample_rate: int = 16000,
        p: float = 0.8,
    ):
        self.tail_ratio_range = tail_ratio_range
        self.speed_factor_range = speed_factor_range
        self.crossfade_ms = crossfade_ms
        self.sample_rate = sample_rate
        self.p = p

    def __call__(
        self,
        waveform: torch.Tensor,
        tail_ratio: Optional[float] = None,
        speed_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply tail speedup to 1D (samples,) or 2D (channels, samples) tensor."""
        if random.random() > self.p and speed_factor is None:
            return waveform

        num_samples = waveform.size(-1)
        # Require at least 0.5s of audio to perform meaningful tail augmentation
        min_samples = int(self.sample_rate * 0.5)
        if num_samples < min_samples:
            return waveform

        if tail_ratio is None:
            tail_ratio = random.uniform(*self.tail_ratio_range)
        if speed_factor is None:
            speed_factor = random.uniform(*self.speed_factor_range)

        if abs(speed_factor - 1.0) < 1e-3:
            return waveform

        split_point = int(num_samples * (1.0 - tail_ratio))
        fade_samples = int(self.sample_rate * (self.crossfade_ms / 1000.0))

        # Ensure both head and tail are longer than cross-fade window
        if split_point <= fade_samples or (num_samples - split_point) <= fade_samples:
            return waveform

        head = waveform[..., :split_point]
        tail = waveform[..., split_point:]

        orig_tail_len = tail.size(-1)
        new_tail_len = max(fade_samples + 1, int(orig_tail_len / speed_factor))

        orig_ndim = tail.ndim
        if orig_ndim == 1:
            tail_3d = tail.view(1, 1, -1)
        elif orig_ndim == 2:
            tail_3d = tail.unsqueeze(0)
        else:
            tail_3d = tail

        fast_tail_3d = F.interpolate(
            tail_3d,
            size=new_tail_len,
            mode="linear",
            align_corners=False,
        )

        if orig_ndim == 1:
            fast_tail = fast_tail_3d.squeeze(0).squeeze(0)
        elif orig_ndim == 2:
            fast_tail = fast_tail_3d.squeeze(0)
        else:
            fast_tail = fast_tail_3d

        # Smooth cross-fading at the boundary to eliminate phase discontinuity & clicking
        if head.size(-1) >= fade_samples and fast_tail.size(-1) >= fade_samples:
            fade_out = torch.linspace(
                1.0, 0.0, fade_samples, device=waveform.device, dtype=waveform.dtype
            )
            fade_in = torch.linspace(
                0.0, 1.0, fade_samples, device=waveform.device, dtype=waveform.dtype
            )

            head_blend = head[..., -fade_samples:] * fade_out
            tail_blend = fast_tail[..., :fade_samples] * fade_in
            blended = head_blend + tail_blend

            augmented = torch.cat(
                [
                    head[..., :-fade_samples],
                    blended,
                    fast_tail[..., fade_samples:],
                ],
                dim=-1,
            )
        else:
            augmented = torch.cat([head, fast_tail], dim=-1)

        return augmented


class TailEnergyDecay:
    """Simulates vocal exhaustion / trailing off by decaying amplitude in the tail.

    Args:
        tail_ratio_range: (min_ratio, max_ratio) fraction of audio treated as tail (default: (0.20, 0.35)).
        decay_db_range: (min_db, max_db) attenuation in decibels (default: (-3.0, -8.0)).
        p: Probability of applying augmentation (default: 0.8).
    """

    def __init__(
        self,
        tail_ratio_range: Tuple[float, float] = (0.20, 0.35),
        decay_db_range: Tuple[float, float] = (-3.0, -8.0),
        p: float = 0.8,
    ):
        self.tail_ratio_range = tail_ratio_range
        self.decay_db_range = decay_db_range
        self.p = p

    def __call__(
        self,
        waveform: torch.Tensor,
        tail_ratio: Optional[float] = None,
        decay_db: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply gradual attenuation ramp to the tail section."""
        if random.random() > self.p and decay_db is None:
            return waveform

        num_samples = waveform.size(-1)
        if num_samples < 1600:
            return waveform

        if tail_ratio is None:
            tail_ratio = random.uniform(*self.tail_ratio_range)
        if decay_db is None:
            decay_db = random.uniform(*self.decay_db_range)

        split_point = int(num_samples * (1.0 - tail_ratio))
        head = waveform[..., :split_point]
        tail = waveform[..., split_point:]
        tail_len = tail.size(-1)

        final_scale = 10.0 ** (decay_db / 20.0)
        gain_ramp = torch.linspace(
            1.0, final_scale, tail_len, device=waveform.device, dtype=waveform.dtype
        )

        decayed_tail = tail * gain_ramp
        return torch.cat([head, decayed_tail], dim=-1)


class TailSpecMasking:
    """Concentrates spectrogram time-frequency masking on the speech tail region.

    Forces the model's acoustic encoder and projector not to over-rely on
    intact spectrogram peaks for utterance-final syllables.

    Args:
        tail_ratio_range: Fraction of the active spectrogram considered tail (default: (0.20, 0.35)).
        freq_mask_param: Maximum frequency channels masked (default: 16).
        time_mask_param: Maximum time frames masked (default: 25).
        mask_count: Number of masks applied in the tail (default: 2).
        mask_value: Value to fill masks (default: 0.0).
        p: Probability of applying augmentation (default: 0.6).
    """

    def __init__(
        self,
        tail_ratio_range: Tuple[float, float] = (0.20, 0.35),
        freq_mask_param: int = 16,
        time_mask_param: int = 25,
        mask_count: int = 2,
        mask_value: float = 0.0,
        p: float = 0.6,
    ):
        self.tail_ratio_range = tail_ratio_range
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.mask_count = mask_count
        self.mask_value = mask_value
        self.p = p

    def __call__(
        self,
        mel_spec: torch.Tensor,
        speech_time_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply targeted masking to the tail portion of mel spectrogram."""
        if random.random() > self.p:
            return mel_spec

        augmented = mel_spec.clone()
        ndim = augmented.ndim

        if ndim == 2:
            n_mels, total_frames = augmented.shape
            # If not specified, look at active frames (for Whisper, up to non-zero or 3000)
            active_frames = speech_time_frames or total_frames
            if active_frames < 30:
                return mel_spec

            tail_ratio = random.uniform(*self.tail_ratio_range)
            tail_start = int(active_frames * (1.0 - tail_ratio))
            tail_duration = max(1, active_frames - tail_start)

            # Frequency masks across active tail
            for _ in range(self.mask_count):
                f = random.randint(1, min(self.freq_mask_param, n_mels))
                f0 = random.randint(0, n_mels - f)
                t_width = random.randint(1, min(self.time_mask_param, tail_duration))
                t0 = tail_start + random.randint(0, tail_duration - t_width)
                augmented[f0 : f0 + f, t0 : t0 + t_width] = self.mask_value

        return augmented


class TailAugmentationPipeline:
    """Unified online augmentation pipeline for tail speech robustness.

    Combines:
    1. Tail Speedup (accelerating ending phonemes by 1.15x - 1.35x)
    2. Tail Energy Decay (linear fade of -3 dB to -8 dB)
    3. Tail Spectrogram Masking
    """

    def __init__(
        self,
        apply_tail_speedup: bool = True,
        tail_speedup_range: Tuple[float, float] = (1.15, 1.35),
        apply_tail_energy_decay: bool = True,
        tail_decay_db_range: Tuple[float, float] = (-3.0, -8.0),
        apply_tail_spec_masking: bool = True,
        tail_ratio_range: Tuple[float, float] = (0.20, 0.35),
        tail_aug_prob: float = 0.8,
        sample_rate: int = 16000,
    ):
        self.tail_speedup = (
            TailSpeedPerturbation(
                tail_ratio_range=tail_ratio_range,
                speed_factor_range=tail_speedup_range,
                sample_rate=sample_rate,
                p=tail_aug_prob,
            )
            if apply_tail_speedup
            else None
        )

        self.tail_energy_decay = (
            TailEnergyDecay(
                tail_ratio_range=tail_ratio_range,
                decay_db_range=tail_decay_db_range,
                p=tail_aug_prob,
            )
            if apply_tail_energy_decay
            else None
        )

        self.tail_spec_masking = (
            TailSpecMasking(
                tail_ratio_range=tail_ratio_range,
                p=tail_aug_prob * 0.75,
            )
            if apply_tail_spec_masking
            else None
        )

    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply active waveform tail augmentations (speedup & decay)."""
        if self.tail_speedup is not None:
            waveform = self.tail_speedup(waveform)
        if self.tail_energy_decay is not None:
            waveform = self.tail_energy_decay(waveform)
        return waveform

    def augment_spectrogram(
        self, mel_spec: torch.Tensor, speech_frames: Optional[int] = None
    ) -> torch.Tensor:
        """Apply active spectrogram tail augmentations."""
        if self.tail_spec_masking is not None:
            mel_spec = self.tail_spec_masking(mel_spec, speech_time_frames=speech_frames)
        return mel_spec
