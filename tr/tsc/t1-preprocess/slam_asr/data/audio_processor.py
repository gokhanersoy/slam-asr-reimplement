"""Audio processing utilities for SLAM-ASR.

Handles loading, resampling to 16 kHz, normalization, and optional
30-second padding & log-Mel spectrogram extraction for Whisper models.
"""

import os
from typing import Optional, Tuple, Union
import torch
import torchaudio
import soundfile as sf
import numpy as np


class AudioProcessor:
    """Processes audio inputs for speech encoders (HuBERT, WavLM, Whisper).

    Supports:
    - Resampling to 16 kHz mono
    - Dynamic Range Compression (DRC) to boost quiet phonemes and trailing speech
    - Leading and trailing silence buffers (acoustic tail guard against truncation)
    - 30-second padding & log-Mel spectrogram extraction for Whisper

    Args:
        sampling_rate: Target sampling rate (default: 16000 Hz).
        normalize: Whether to normalize audio waveform to zero mean and unit variance.
        is_whisper: Whether the target encoder is Whisper (requires 80/128 mel spectrogram & 30s padding).
        whisper_mel_bins: Number of mel bins for Whisper (default: 80 for whisper-large-v2, 128 for v3).
        apply_drc: Whether to apply soft dynamic range compression to boost quiet trailing phonemes.
        drc_threshold_db: Dynamic range compression threshold in dB (default: -24.0 dB).
        drc_ratio: Dynamic range compression ratio above threshold (default: 3.0).
        trailing_silence_sec: Seconds of neutral silence padding added at the end of speech (default: 0.3s).
        leading_silence_sec: Seconds of neutral silence padding added at the beginning (default: 0.1s).
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        normalize: bool = False,
        is_whisper: bool = False,
        whisper_mel_bins: int = 80,
        apply_drc: bool = True,
        drc_threshold_db: float = -24.0,
        drc_ratio: float = 3.0,
        trailing_silence_sec: float = 0.3,
        leading_silence_sec: float = 0.1,
    ):
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.is_whisper = is_whisper
        self.whisper_mel_bins = whisper_mel_bins
        self.apply_drc = apply_drc
        self.drc_threshold_db = drc_threshold_db
        self.drc_ratio = drc_ratio
        self.trailing_silence_sec = trailing_silence_sec
        self.leading_silence_sec = leading_silence_sec

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

    def apply_drc_compression(self, waveform: torch.Tensor) -> torch.Tensor:
        """Smooth envelope dynamic range compression using moving RMS in PyTorch.

        Boosts quiet trailing syllables and compensates for subglottal pressure drop.
        """
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        frame_len = int(self.sampling_rate * 0.025)  # 25 ms frame
        if frame_len % 2 == 0:
            frame_len += 1
        pad_len = frame_len // 2

        wave_2d = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        padded = torch.nn.functional.pad(wave_2d**2, (pad_len, pad_len), mode="reflect")
        rms = torch.sqrt(torch.nn.functional.avg_pool1d(padded, kernel_size=frame_len, stride=1).squeeze() + 1e-7)

        max_rms = rms.max().clamp(min=1e-7)
        rms_db = 20.0 * torch.log10((rms / max_rms).clamp(min=1e-5))

        gain_db = torch.zeros_like(rms_db)
        mask = rms_db > self.drc_threshold_db
        # Downward compression: attenuate peaks above threshold
        gain_db[mask] = (self.drc_threshold_db + (rms_db[mask] - self.drc_threshold_db) / self.drc_ratio) - rms_db[mask]

        # Apply gain reduction (gain <= 1.0)
        gain = 10.0 ** (gain_db / 20.0)
        compressed = waveform * gain

        # Make-up gain / peak normalization: boosts quiet trailing speech
        peak = torch.max(torch.abs(compressed)).clamp(min=1e-7)
        compressed = compressed * (0.95 / peak)
        return compressed

    def process(self, audio_path_or_array: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process a single audio waveform with dynamic range compression and tail buffering.

        For HuBERT / WavLM:
            Returns raw 1D waveform tensor of shape (samples,).
        For Whisper:
            Pads/trims to 30s and computes log-mel spectrogram of shape (n_mels, 3000).
        """
        waveform = self.load_audio(audio_path_or_array)

        # 1. Apply Dynamic Range Compression to boost quiet trailing speech
        if self.apply_drc:
            waveform = self.apply_drc_compression(waveform)

        # 2. Add Leading and Trailing Silence Buffers (acoustic tail guard against truncation)
        pad_left = int(self.sampling_rate * self.leading_silence_sec) if self.leading_silence_sec > 0 else 0
        pad_right = int(self.sampling_rate * self.trailing_silence_sec) if self.trailing_silence_sec > 0 else 0
        if pad_left > 0 or pad_right > 0:
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), value=0.0)

        # 3. Optional waveform normalization
        if self.normalize:
            # Zero mean and unit variance normalization
            mean = waveform.mean()
            std = waveform.std().clamp(min=1e-7)
            waveform = (waveform - mean) / std

        if not self.is_whisper:
            return waveform

        # 4. Whisper pipeline: pad to 30 seconds
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

        return log_mel_spec
