"""Audio processing utilities for SLAM-ASR.

Handles loading, resampling to 16 kHz, normalization, and optional
30-second padding & log-Mel spectrogram extraction for Whisper models.
"""

from typing import Optional, Tuple, Union
import torch
import torchaudio
import soundfile as sf
import numpy as np


class AudioProcessor:
    """Processes audio inputs for speech encoders (HuBERT, WavLM, Whisper).

    Args:
        sampling_rate: Target sampling rate (default: 16000 Hz).
        normalize: Whether to normalize audio waveform to zero mean and unit variance.
        is_whisper: Whether the target encoder is Whisper (requires 80/128 mel spectrogram & 30s padding).
        whisper_mel_bins: Number of mel bins for Whisper (default: 80 for whisper-large-v2, 128 for v3).
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        normalize: bool = False,
        is_whisper: bool = False,
        whisper_mel_bins: int = 80,
    ):
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.is_whisper = is_whisper
        self.whisper_mel_bins = whisper_mel_bins

        # Whisper standard constants
        self.n_fft = 400
        self.hop_length = 160
        self.whisper_max_length_samples = 30 * sampling_rate  # 30 seconds = 480,000 samples

    def load_audio(self, audio_path_or_array: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Load audio from file path or convert from numpy/tensor to 1D float tensor at 16 kHz."""
        if isinstance(audio_path_or_array, str):
            waveform, sr = torchaudio.load(audio_path_or_array)
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

    def process(self, audio_path_or_array: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process a single audio waveform.

        For HuBERT / WavLM:
            Returns raw 1D waveform tensor of shape (samples,).
        For Whisper:
            Pads/trims to 30s and computes log-mel spectrogram of shape (n_mels, 3000).
        """
        waveform = self.load_audio(audio_path_or_array)

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

        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=0.0,
            f_max=8000.0,
            n_mels=self.whisper_mel_bins,
            sample_rate=self.sampling_rate,
        ).transpose(0, 1)

        mel_spec = torch.matmul(mel_filters, magnitudes)
        log_mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel_spec = torch.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
        log_mel_spec = (log_mel_spec + 4.0) / 4.0

        return log_mel_spec
