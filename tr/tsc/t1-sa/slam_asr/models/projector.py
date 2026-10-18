"""Projector modules for SLAM-ASR (arXiv:2402.08846) with Spectral Attention extensions.

The projector acts as a bridge between the speech encoder and the LLM.
It downsamples speech features temporally and maps them to the LLM's embedding space.

Supported Projector Types:
1. 'linear': Direct concatenation of k frames + Single Linear projection (SLAM-ASR baseline).
2. 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
3. 'spectral_attention' (or 'spectral', 'sa'): Cross-Subband Spectral Self-Attention
   (harmonic & formant modeling across frequency channels) + Dynamic Spectral Channel Gating
   + Temporal Downsampling + Base Projection.
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameDownsampler(nn.Module):
    """Downsamples speech features along the temporal dimension by concatenating
    every k consecutive frames in the feature dimension.

    Args:
        k: Downsampling factor (default: 5, reducing 50 Hz speech frames to 10 Hz).
        padding_mode: How to handle sequences whose length is not divisible by k.
            'discard': Discard the trailing frames (official SLAM-LLM implementation).
            'pad': Zero-pad the sequence at the end so no frames are dropped.
    """

    def __init__(self, k: int = 5, padding_mode: Literal["discard", "pad"] = "discard"):
        super().__init__()
        self.k = k
        self.padding_mode = padding_mode

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Downsample sequence from (batch_size, seq_len, dim) to
        (batch_size, seq_len // k, dim * k).

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            attention_mask: Optional binary mask of shape (batch, seq_len).

        Returns:
            Tuple of (downsampled_features, downsampled_mask).
        """
        batch_size, seq_len, dim = x.size()
        remainder = seq_len % self.k

        if remainder > 0:
            if self.padding_mode == "discard":
                x = x[:, :-remainder, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :-remainder]
            elif self.padding_mode == "pad":
                pad_len = self.k - remainder
                x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
                if attention_mask is not None:
                    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

        new_seq_len = x.size(1) // self.k
        downsampled = x.contiguous().view(batch_size, new_seq_len, dim * self.k)

        downsampled_mask = None
        if attention_mask is not None:
            # Mask frame is valid if at least one sub-frame was valid
            downsampled_mask = (
                attention_mask.view(batch_size, new_seq_len, self.k).sum(dim=-1) > 0
            ).long()

        return downsampled, downsampled_mask


class LinearProjector(nn.Module):
    """Single linear projector for SLAM-ASR.

    Maps downsampled speech features directly to LLM hidden dimension.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProjector(nn.Module):
    """Two-layer MLP projector for SLAM-ASR as described in Section 2.3 & 3.1.3:
    Linear(in_dim, hidden_dim) -> ReLU() -> Linear(hidden_dim, out_dim).
    Paper defaults: hidden_dim = 2048.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class SpectralSubBandAttention(nn.Module):
    """Cross-Subband Spectral Self-Attention.

    Divides the continuous feature dimension D into S spectral sub-bands.
    Applies multi-head self-attention across the spectral sub-bands at each time frame,
    modeling non-local harmonic correlations (fundamental frequency F0, overtones,
    formant F1-F2 relationships, and vowel-consonant spectral transitions).

    Args:
        dim: Acoustic feature dimension D (e.g. 1280).
        num_bands: Number of spectral sub-bands S (default: 16).
        num_heads: Number of attention heads across spectral bands (default: 4).
        dropout: Attention dropout probability (default: 0.1).
    """

    def __init__(
        self,
        dim: int,
        num_bands: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim % num_bands != 0:
            raise ValueError(f"Feature dimension ({dim}) must be divisible by num_bands ({num_bands})")
        self.dim = dim
        self.num_bands = num_bands
        self.band_dim = dim // num_bands

        self.norm1 = nn.LayerNorm(self.band_dim)
        self.spectral_attn = nn.MultiheadAttention(
            embed_dim=self.band_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch_size, time_frames, dim)

        Returns:
            out: Spectrally attended features (batch_size, time_frames, dim)
        """
        batch_size, time_frames, dim = x.size()

        # Reshape into spectral sub-bands: (B * T, S, band_dim)
        x_bands = x.view(batch_size * time_frames, self.num_bands, self.band_dim)
        norm_bands = self.norm1(x_bands)

        # Cross-subband attention across frequency channels
        attn_out, _ = self.spectral_attn(
            query=norm_bands,
            key=norm_bands,
            value=norm_bands,
        )
        attn_out = self.dropout(attn_out)

        # Reshape back to (B, T, D)
        spectral_features = attn_out.contiguous().view(batch_size, time_frames, dim)

        # Gated residual connection + LayerNorm
        out = self.norm2(x + torch.tanh(self.gate) * spectral_features)
        return out


class SpectralChannelGating(nn.Module):
    """Dynamic Spectral Channel Squeeze-and-Excitation.

    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between spectral channels. Dynamically emphasizes
    salient phonetic frequency bands (formants, fricative noise bands) while
    suppressing uninformative ambient noise channels.

    Args:
        dim: Feature dimension D (e.g. 1280).
        reduction: Channel bottleneck reduction ratio (default: 4).
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(dim // reduction, 32)
        self.fc = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x: (batch_size, time_frames, dim)
            attention_mask: Optional mask (batch_size, time_frames)

        Returns:
            Modulated features (batch_size, time_frames, dim)
        """
        if attention_mask is not None:
            # Masked average pooling across time
            mask_expanded = attention_mask.unsqueeze(-1).float()
            time_avg = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            time_avg = x.mean(dim=1)  # (B, D)

        # Spectral channel modulation weights: (B, 1, D) in range (0, 1)
        channel_weights = self.fc(time_avg).unsqueeze(1)
        # Scale input features
        return x * channel_weights


class SpectralAttentionProjector(nn.Module):
    """Spectral Attention Projector for SLAM-ASR (arXiv:2402.08846 + SA extension).

    Architecture Pipeline:
    1. Cross-Subband Spectral Self-Attention: Learns harmonic and formant correlations across frequency channels.
    2. Dynamic Spectral Channel Gating: Recalibrates channel-wise responses to suppress spectral noise.
    3. Temporal Downsampler: k=5 consecutive frame concatenation (50 Hz -> 10 Hz).
    4. Base Projection: Linear projection from concatenated dimension to LLM embedding dimension.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        num_spectral_bands: int = 16,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        projector_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
    ):
        super().__init__()
        self.spectral_subband_attn = SpectralSubBandAttention(
            dim=encoder_dim,
            num_bands=num_spectral_bands,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )
        self.spectral_channel_gating = SpectralChannelGating(
            dim=encoder_dim,
            reduction=4,
        )
        self.downsampler = FrameDownsampler(k=downsample_rate, padding_mode=padding_mode)
        concatenated_dim = encoder_dim * downsample_rate

        if projector_type == "linear":
            self.base_projector = LinearProjector(concatenated_dim, llm_dim)
        elif projector_type == "mlp":
            self.base_projector = MLPProjector(concatenated_dim, llm_dim, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported base projector_type: {projector_type}")

    def forward(
        self,
        speech_features: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Cross-subband spectral self-attention across frequency channels
        spec_features = self.spectral_subband_attn(speech_features)
        # 2. Dynamic spectral channel gating (noise suppression & formant boost)
        gated_features = self.spectral_channel_gating(spec_features, speech_attention_mask)
        # 3. Temporal downsampling (k=5 concatenation)
        downsampled, mask = self.downsampler(gated_features, speech_attention_mask)
        # 4. Base projection into LLM space
        projected = self.base_projector(downsampled)
        return projected, mask


class SpeechProjector(nn.Module):
    """Unified Speech Projector for SLAM-ASR.

    Supported `projector_type` options:
        - 'linear': Direct concatenation of k frames + Linear layer (SLAM-ASR baseline).
        - 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
        - 'spectral_attention' (or 'spectral', 'sa'): Cross-Subband Spectral Self-Attention
          + Dynamic Spectral Channel Gating + Temporal Downsampling + Projection.

    Args:
        encoder_dim: Dimension of speech encoder output features (e.g. 1280 for Whisper-Large-v3).
        llm_dim: Dimension of LLM input embeddings (e.g. 3584 for Qwen2.5-7B).
        downsample_rate: Downsampling factor k (default: 5).
        projector_type: 'linear', 'mlp', or 'spectral_attention' (default: 'spectral_attention').
        hidden_dim: Hidden dimension when projector_type='mlp' (default: 2048).
        padding_mode: 'discard' or 'pad' for downsampler remainder.
        num_spectral_bands: Number of sub-bands for spectral attention (default: 16).
        num_attention_heads: Number of attention heads (default: 4).
        attention_dropout: Dropout probability for attention layers (default: 0.1).
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        projector_type: Literal["linear", "mlp", "spectral_attention", "spectral", "sa"] = "spectral_attention",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
        num_spectral_bands: int = 16,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.downsample_rate = downsample_rate
        self.projector_type = projector_type

        if projector_type in ("spectral_attention", "spectral", "sa"):
            self.impl = SpectralAttentionProjector(
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
                downsample_rate=downsample_rate,
                num_spectral_bands=num_spectral_bands,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                projector_type="linear",
                hidden_dim=hidden_dim,
                padding_mode=padding_mode,
            )
        else:
            self.downsampler = FrameDownsampler(k=downsample_rate, padding_mode=padding_mode)
            concatenated_dim = encoder_dim * downsample_rate
            if projector_type == "linear":
                self.projector = LinearProjector(concatenated_dim, llm_dim)
            elif projector_type == "mlp":
                self.projector = MLPProjector(concatenated_dim, llm_dim, hidden_dim=hidden_dim)
            else:
                raise ValueError(
                    f"Unsupported projector_type: {projector_type}. "
                    f"Choose 'linear', 'mlp', or 'spectral_attention'."
                )
            self.impl = None

    def forward(
        self,
        speech_features: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            speech_features: (batch_size, time_frames, encoder_dim)
            speech_attention_mask: Optional mask (batch_size, time_frames)

        Returns:
            speech_embeds: (batch_size, downsampled_time, llm_dim)
            downsampled_mask: (batch_size, downsampled_time) or None
        """
        if self.impl is not None:
            return self.impl(speech_features, speech_attention_mask)
        downsampled, mask = self.downsampler(speech_features, speech_attention_mask)
        projected = self.projector(downsampled)
        return projected, mask
