"""Audio and Spectrogram Data Augmentations for SLAM-ASR (t1-feature-aug).

Implements:
1. Speed Perturbation: 3-way speed variation (0.9x, 1.0x, 1.1x) to model speech rate variability.
2. Volume Perturbation: Random gain jitter (-4 dB to +4 dB) to handle acoustic energy diversity.
3. SpecAugment (Park et al., 2019): Time and frequency masking on log-Mel spectrograms
   to force robust phonetic context modeling and prevent overfitting on speaker timbre.
"""

import random
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn


class SpeedPerturbation:
    """Waveform speed perturbation via linear resampling.

    Simulates speech rate variations (slower/faster speech, acceleration at phrase boundaries).

    Args:
        factors: Allowed speed factors (e.g. [0.9, 1.0, 1.1]).
    """

    def __init__(self, factors: Sequence[float] = (0.9, 1.0, 1.1)):
        self.factors = list(factors)

    def __call__(self, waveform: torch.Tensor, factor: Optional[float] = None) -> torch.Tensor:
        """Apply speed perturbation to 1D waveform."""
        if factor is None:
            factor = random.choice(self.factors)

        if abs(factor - 1.0) < 1e-4:
            return waveform

        orig_ndim = waveform.ndim
        if orig_ndim == 1:
            wave_3d = waveform.view(1, 1, -1)
        elif orig_ndim == 2:
            wave_3d = waveform.unsqueeze(0)
        else:
            wave_3d = waveform

        old_len = wave_3d.size(-1)
        new_len = max(1, int(old_len / factor))

        resampled = torch.nn.functional.interpolate(
            wave_3d,
            size=new_len,
            mode="linear",
            align_corners=False,
        )

        if orig_ndim == 1:
            return resampled.squeeze(0).squeeze(0)
        elif orig_ndim == 2:
            return resampled.squeeze(0)
        return resampled


class VolumePerturbation:
    """Random volume/gain perturbation in dB.

    Args:
        min_db: Minimum gain change in dB (default: -4.0).
        max_db: Maximum gain change in dB (default: +4.0).
    """

    def __init__(self, min_db: float = -4.0, max_db: float = 4.0):
        self.min_db = min_db
        self.max_db = max_db

    def __call__(self, waveform: torch.Tensor, db_shift: Optional[float] = None) -> torch.Tensor:
        """Apply random gain shift."""
        if db_shift is None:
            db_shift = random.uniform(self.min_db, self.max_db)

        if abs(db_shift) < 1e-4:
            return waveform

        scale = 10.0 ** (db_shift / 20.0)
        return torch.clamp(waveform * scale, min=-1.0, max=1.0)


class SpecAugment:
    """SpecAugment (Park et al., 2019) time and frequency masking on spectrograms.

    Args:
        freq_mask_param: Maximum width of frequency mask F.
        time_mask_param: Maximum width of time mask T.
        freq_mask_count: Number of frequency masks to apply.
        time_mask_count: Number of time masks to apply.
        mask_value: Value used to fill masked regions (default: 0.0).
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        freq_mask_count: int = 2,
        time_mask_count: int = 2,
        mask_value: float = 0.0,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.freq_mask_count = freq_mask_count
        self.time_mask_count = time_mask_count
        self.mask_value = mask_value

    def __call__(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to 2D (n_mels, time) or 3D (batch, n_mels, time) tensor."""
        augmented = mel_spec.clone()

        if augmented.ndim == 2:
            n_mels, time_steps = augmented.shape
            # Frequency masking
            for _ in range(self.freq_mask_count):
                f = random.randint(0, min(self.freq_mask_param, n_mels))
                if f > 0:
                    f0 = random.randint(0, n_mels - f)
                    augmented[f0 : f0 + f, :] = self.mask_value

            # Time masking
            for _ in range(self.time_mask_count):
                t = random.randint(0, min(self.time_mask_param, time_steps))
                if t > 0:
                    t0 = random.randint(0, time_steps - t)
                    augmented[:, t0 : t0 + t] = self.mask_value

        elif augmented.ndim == 3:
            batch, n_mels, time_steps = augmented.shape
            for b in range(batch):
                for _ in range(self.freq_mask_count):
                    f = random.randint(0, min(self.freq_mask_param, n_mels))
                    if f > 0:
                        f0 = random.randint(0, n_mels - f)
                        augmented[b, f0 : f0 + f, :] = self.mask_value

                for _ in range(self.time_mask_count):
                    t = random.randint(0, min(self.time_mask_param, time_steps))
                    if t > 0:
                        t0 = random.randint(0, time_steps - t)
                        augmented[b, :, t0 : t0 + t] = self.mask_value

        return augmented


class AudioAugmentationPipeline:
    """Unified audio augmentation pipeline combining waveform and spectrogram perturbations.

    Args:
        apply_speed_perturbation: Whether to apply speed variation.
        speed_factors: Allowed speed factors (default: [0.9, 1.0, 1.1]).
        apply_volume_perturbation: Whether to apply random gain shift.
        volume_db_range: (min_db, max_db) tuple (default: (-4.0, 4.0)).
        apply_spec_augment: Whether to apply frequency and time masking on spectrograms.
        freq_mask_param: Max frequency mask size (default: 15).
        time_mask_param: Max time mask size (default: 35).
        freq_mask_count: Number of frequency masks (default: 2).
        time_mask_count: Number of time masks (default: 2).
    """

    def __init__(
        self,
        apply_speed_perturbation: bool = True,
        speed_factors: Sequence[float] = (0.9, 1.0, 1.1),
        apply_volume_perturbation: bool = True,
        volume_db_range: Tuple[float, float] = (-4.0, 4.0),
        apply_spec_augment: bool = True,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        freq_mask_count: int = 2,
        time_mask_count: int = 2,
    ):
        self.apply_speed_perturbation = apply_speed_perturbation
        self.speed_perturb = SpeedPerturbation(factors=speed_factors) if apply_speed_perturbation else None

        self.apply_volume_perturbation = apply_volume_perturbation
        self.volume_perturb = (
            VolumePerturbation(min_db=volume_db_range[0], max_db=volume_db_range[1])
            if apply_volume_perturbation
            else None
        )

        self.apply_spec_augment = apply_spec_augment
        self.spec_augment = (
            SpecAugment(
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                freq_mask_count=freq_mask_count,
                time_mask_count=time_mask_count,
            )
            if apply_spec_augment
            else None
        )

    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply active waveform augmentations (speed & volume)."""
        if self.speed_perturb is not None:
            waveform = self.speed_perturb(waveform)
        if self.volume_perturb is not None:
            waveform = self.volume_perturb(waveform)
        return waveform

    def augment_spectrogram(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply active spectrogram augmentations (SpecAugment)."""
        if self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)
        return mel_spec
