"""Audio processing utilities for SLAM-ASR.

Handles loading, resampling to 16 kHz, normalization, and optional
30-second padding & log-Mel spectrogram extraction for Whisper models.
"""

import os
from typing import Optional, Sequence, Tuple, Union
import torch
import torchaudio
import soundfile as sf
import numpy as np

from slam_asr.data.augmentations import AudioAugmentationPipeline


class AudioProcessor:
    """Processes audio inputs for speech encoders (HuBERT, WavLM, Whisper).

    Supports optional online acoustic augmentations during training:
    - Speed Perturbation (e.g. 0.9x, 1.0x, 1.1x)
    - Volume / Gain Perturbation (-4 dB to +4 dB)
    - SpecAugment (Time and Frequency Masking on Log-Mel Spectrograms)

    Args:
        sampling_rate: Target sampling rate (default: 16000 Hz).
        normalize: Whether to normalize audio waveform to zero mean and unit variance.
        is_whisper: Whether the target encoder is Whisper (requires 80/128 mel spectrogram & 30s padding).
        whisper_mel_bins: Number of mel bins for Whisper (default: 80 for whisper-large-v2, 128 for v3).
        apply_speed_perturbation: Whether to apply waveform speed perturbation.
        speed_factors: Speed perturbation ratios (default: [0.9, 1.0, 1.1]).
        apply_volume_perturbation: Whether to apply random gain jitter.
        volume_db_range: Minimum and maximum gain in dB (default: (-4.0, 4.0)).
        apply_spec_augment: Whether to apply SpecAugment time and frequency masking.
        freq_mask_param: Maximum frequency channels masked (default: 15).
        time_mask_param: Maximum time frames masked (default: 35).
        freq_mask_count: Number of frequency masks (default: 2).
        time_mask_count: Number of time masks (default: 2).
        is_training: If True, augmentations are active; if False, disabled (default: False).
        augmentation_pipeline: Optional preconfigured AudioAugmentationPipeline.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        normalize: bool = False,
        is_whisper: bool = False,
        whisper_mel_bins: int = 80,
        apply_speed_perturbation: bool = False,
        speed_factors: Sequence[float] = (0.9, 1.0, 1.1),
        apply_volume_perturbation: bool = False,
        volume_db_range: Tuple[float, float] = (-4.0, 4.0),
        apply_spec_augment: bool = False,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        freq_mask_count: int = 2,
        time_mask_count: int = 2,
        is_training: bool = False,
        augmentation_pipeline: Optional[AudioAugmentationPipeline] = None,
    ):
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.is_whisper = is_whisper
        self.whisper_mel_bins = whisper_mel_bins
        self.is_training = is_training

        # Initialize augmentation pipeline if enabled
        if augmentation_pipeline is not None:
            self.augmentation_pipeline = augmentation_pipeline
        elif apply_speed_perturbation or apply_volume_perturbation or apply_spec_augment:
            self.augmentation_pipeline = AudioAugmentationPipeline(
                apply_speed_perturbation=apply_speed_perturbation,
                speed_factors=speed_factors,
                apply_volume_perturbation=apply_volume_perturbation,
                volume_db_range=volume_db_range,
                apply_spec_augment=apply_spec_augment,
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                freq_mask_count=freq_mask_count,
                time_mask_count=time_mask_count,
            )
        else:
            self.augmentation_pipeline = None

        # Whisper standard constants
        self.n_fft = 400
        self.hop_length = 160
        self.whisper_max_length_samples = 30 * sampling_rate  # 30 seconds = 480,000 samples

        # Precompute Mel filterbank matrix once for Whisper (Slaney norm matches official Whisper)
        try:
            self.mel_filters = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=0.0,
                f_max=8000.0,
                n_mels=self.whisper_mel_bins,
                sample_rate=self.sampling_rate,
                norm="slaney",
                mel_scale="slaney",
            ).transpose(0, 1)
        except Exception:
            self.mel_filters = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=0.0,
                f_max=8000.0,
                n_mels=self.whisper_mel_bins,
                sample_rate=self.sampling_rate,
            ).transpose(0, 1)

    def load_audio(self, audio_path_or_array: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Load audio from file path or convert from numpy/tensor to 1D float tensor at 16 kHz."""
        if isinstance(audio_path_or_array, str):
            resolved_path = audio_path_or_array
            if not os.path.exists(resolved_path):
                # Handle cross-machine absolute paths (e.g. from local Mac into Colab)
                if "data/" in resolved_path:
                    candidate = resolved_path[resolved_path.find("data/"):]
                    if os.path.exists(candidate):
                        resolved_path = candidate
                if not os.path.exists(resolved_path):
                    filename = os.path.basename(resolved_path)
                    for folder in ["data/toy_audio", "data", "data/colab_audio"]:
                        candidate = os.path.join(folder, filename)
                        if os.path.exists(candidate):
                            resolved_path = candidate
                            break

            waveform, sr = torchaudio.load(resolved_path)
            # Convert multi-channel to mono
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Resample if needed
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                waveform = resampler(waveform)
            waveform = waveform.squeeze(0)  # (samples,)
        elif isinstance(audio_path_or_array, np.ndarray):
            waveform = torch.from_numpy(audio_path_or_array).float()
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=-1)
        elif isinstance(audio_path_or_array, torch.Tensor):
            waveform = audio_path_or_array.float()
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=-1)
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_path_or_array)}")

        return waveform

    def train(self, mode: bool = True):
        """Set training mode (enables augmentations)."""
        self.is_training = mode
        return self

    def eval(self):
        """Set evaluation mode (disables augmentations)."""
        self.is_training = False
        return self

    def process(self, audio_path_or_array: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process a single audio waveform with optional data augmentations.

        For HuBERT / WavLM:
            Returns raw 1D waveform tensor of shape (samples,).
        For Whisper:
            Pads/trims to 30s and computes log-mel spectrogram of shape (n_mels, 3000).
        """
        waveform = self.load_audio(audio_path_or_array)

        # 1. Waveform augmentations (speed & volume perturbation during training)
        if self.is_training and self.augmentation_pipeline is not None:
            waveform = self.augmentation_pipeline.augment_waveform(waveform)

        if self.normalize:
            # Zero mean and unit variance normalization
            mean = waveform.mean()
            std = waveform.std().clamp(min=1e-7)
            waveform = (waveform - mean) / std

        if not self.is_whisper:
            return waveform

        # Whisper pipeline: pad to 30 seconds
        samples = waveform.size(0)
        if samples < self.whisper_max_length_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.whisper_max_length_samples - samples), value=0.0
            )
        else:
            waveform = waveform[: self.whisper_max_length_samples]

        # Extract log-mel spectrogram using torchaudio
        window = torch.hann_window(self.n_fft)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = self.mel_filters.to(magnitudes.device)
        mel_spec = torch.matmul(mel_filters, magnitudes)
        log_mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel_spec = torch.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
        log_mel_spec = (log_mel_spec + 4.0) / 4.0

        # 2. Spectrogram augmentations (SpecAugment frequency and time masking during training)
        if self.is_training and self.augmentation_pipeline is not None:
            log_mel_spec = self.augmentation_pipeline.augment_spectrogram(log_mel_spec)

        return log_mel_spec
