"""Projector modules for SLAM-ASR (arXiv:2402.08846) with Temporal Attention extensions.

The projector acts as a bridge between the speech encoder and the LLM.
It downsamples speech features temporally and maps them to the LLM's embedding space.

Supported Projector Types:
1. 'linear': Direct concatenation of k frames + Single Linear projection (SLAM-ASR baseline).
2. 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
3. 'temporal_attention': Intra-window Temporal Attention Pooling (TSN-style)
   + Linear/MLP projection + Inter-window Multi-Head Temporal Self-Attention.
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


class TemporalAttentionPooling(nn.Module):
    """Segment-level / Intra-window Temporal Attention Pooling (TSN-style).

    Given speech features grouped into temporal windows of k frames:
        x_window: (batch_size, num_windows, k, dim)
    Computes a learned dynamic attention score for each frame within the window:
        scores = v^T tanh(W x + b)
        weights = softmax(scores, dim=-2)
    And outputs the attention-weighted temporal centroid:
        attended = sum(weights * x_window, dim=-2)
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        attn_dim = hidden_dim or max(dim // 4, 64)
        self.proj = nn.Sequential(
            nn.Linear(dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )

    def forward(self, x_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x_window: (batch_size, num_windows, k, dim)

        Returns:
            Tuple of:
                attended: (batch_size, num_windows, dim)
                weights: (batch_size, num_windows, k)
        """
        scores = self.proj(x_window)  # (batch_size, num_windows, k, 1)
        weights = F.softmax(scores, dim=-2)  # (batch_size, num_windows, k, 1)
        attended = (x_window * weights).sum(dim=-2)  # (batch_size, num_windows, dim)
        return attended, weights.squeeze(-1)


class TemporalAttentionDownsampler(nn.Module):
    """Downsamples speech features along the temporal dimension using a combination
    of window segmentation and Temporal Attention.

    Args:
        k: Downsampling factor (default: 5).
        encoder_dim: Dimension of speech features.
        pooling_mode:
            - 'hybrid': Concatenates the k frames AND appends the temporal attention centroid.
                        Output dimension = (k + 1) * encoder_dim.
            - 'attention': Replaces concatenation with temporal attention pooling.
                           Output dimension = encoder_dim.
            - 'weighted_concat': Multiplies each frame by its learned temporal attention
                                 weight before concatenation. Output dimension = k * encoder_dim.
        padding_mode: 'discard' or 'pad'.
    """

    def __init__(
        self,
        k: int = 5,
        encoder_dim: int = 1280,
        pooling_mode: Literal["hybrid", "attention", "weighted_concat"] = "hybrid",
        padding_mode: Literal["discard", "pad"] = "discard",
    ):
        super().__init__()
        self.k = k
        self.encoder_dim = encoder_dim
        self.pooling_mode = pooling_mode
        self.padding_mode = padding_mode

        self.attn_pooling = TemporalAttentionPooling(encoder_dim)

        if pooling_mode == "hybrid":
            self.out_dim = (k + 1) * encoder_dim
        elif pooling_mode == "attention":
            self.out_dim = encoder_dim
        elif pooling_mode == "weighted_concat":
            self.out_dim = k * encoder_dim
        else:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        x_windows = x.contiguous().view(batch_size, new_seq_len, self.k, dim)

        attended, weights = self.attn_pooling(x_windows)

        if self.pooling_mode == "hybrid":
            concat = x_windows.view(batch_size, new_seq_len, dim * self.k)
            downsampled = torch.cat([concat, attended], dim=-1)
        elif self.pooling_mode == "attention":
            downsampled = attended
        elif self.pooling_mode == "weighted_concat":
            weighted_x = x_windows * weights.unsqueeze(-1)
            downsampled = weighted_x.contiguous().view(batch_size, new_seq_len, dim * self.k)

        downsampled_mask = None
        if attention_mask is not None:
            downsampled_mask = (
                attention_mask.view(batch_size, new_seq_len, self.k).sum(dim=-1) > 0
            ).long()

        return downsampled, downsampled_mask


class TemporalSelfAttentionBlock(nn.Module):
    """Multi-Head Temporal Self-Attention over downsampled speech representations.

    Allows downsampled acoustic tokens to attend across the entire utterance,
    modeling long-range phonotactic patterns and morphological vowel harmony.
    Uses Pre-LayerNorm architecture with residual connections and FFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ffn_dim = ffn_dim or (dim * 2)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        attn_out, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalAttentionProjector(nn.Module):
    """Temporal Attention Projector for SLAM-ASR (arXiv:2402.08846 + TA extension).

    Combines:
    1. Segmental Temporal Attention Downsampling (intra-window dynamic weighting)
    2. Linear or MLP Projection into LLM embedding space
    3. Multi-Head Temporal Self-Attention (inter-window global temporal modeling)
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        pooling_mode: Literal["hybrid", "attention", "weighted_concat"] = "hybrid",
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        projector_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
    ):
        super().__init__()
        self.downsampler = TemporalAttentionDownsampler(
            k=downsample_rate,
            encoder_dim=encoder_dim,
            pooling_mode=pooling_mode,
            padding_mode=padding_mode,
        )
        downsampled_dim = self.downsampler.out_dim

        if projector_type == "linear":
            self.base_projector = LinearProjector(downsampled_dim, llm_dim)
        elif projector_type == "mlp":
            self.base_projector = MLPProjector(downsampled_dim, llm_dim, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported base projector_type: {projector_type}")

        self.temporal_self_attn = TemporalSelfAttentionBlock(
            dim=llm_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )

    def forward(
        self,
        speech_features: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        downsampled, mask = self.downsampler(speech_features, speech_attention_mask)
        projected = self.base_projector(downsampled)
        out = self.temporal_self_attn(projected, mask)
        return out, mask


class SpeechProjector(nn.Module):
    """Unified Speech Projector for SLAM-ASR.

    Combines temporal downsampling with projection into the LLM embedding space.

    Supported `projector_type` options:
        - 'linear': Direct concatenation of k frames + Linear layer (SLAM-ASR baseline).
        - 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
        - 'temporal_attention' (or 'ta'): Temporal Attention Downsampler (TSN-style)
          + Linear projection + Multi-Head Temporal Self-Attention over speech tokens.

    Args:
        encoder_dim: Dimension of speech encoder output features (e.g. 1280 for Whisper-Large-v3).
        llm_dim: Dimension of LLM input embeddings (e.g. 3584 for Qwen2.5-7B).
        downsample_rate: Downsampling factor k (default: 5).
        projector_type: 'linear', 'mlp', or 'temporal_attention' (default: 'temporal_attention').
        hidden_dim: Hidden dimension when projector_type='mlp' (default: 2048).
        padding_mode: 'discard' or 'pad' for downsampler remainder.
        temporal_pooling_mode: 'hybrid', 'attention', or 'weighted_concat' (default: 'hybrid').
        num_attention_heads: Number of attention heads for Temporal Self-Attention (default: 4).
        attention_dropout: Dropout probability for Temporal Attention (default: 0.1).
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        projector_type: Literal["linear", "mlp", "temporal_attention", "ta"] = "temporal_attention",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
        temporal_pooling_mode: Literal["hybrid", "attention", "weighted_concat"] = "hybrid",
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.downsample_rate = downsample_rate
        self.projector_type = projector_type

        if projector_type in ("temporal_attention", "temporal", "ta"):
            self.impl = TemporalAttentionProjector(
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
                downsample_rate=downsample_rate,
                pooling_mode=temporal_pooling_mode,
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
                    f"Choose 'linear', 'mlp', or 'temporal_attention'."
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
