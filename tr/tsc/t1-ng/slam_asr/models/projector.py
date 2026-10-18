"""Projector modules for SLAM-ASR (arXiv:2402.08846) with N-Gram Attention extensions.

The projector acts as a bridge between the speech encoder and the LLM.
It downsamples speech features temporally and maps them to the LLM's embedding space.

Supported Projector Types:
1. 'linear': Direct concatenation of k frames + Single Linear projection (SLAM-ASR baseline).
2. 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
3. 'ngram_attention' (or 'ngram', 'ng'): Multi-Order Acoustic N-Gram Convolutions
   (1-gram, 2-gram, 3-gram, 5-gram coarticulation context) + Temporal Downsampling
   + Base Projection + Local Windowed N-Gram Contextual Self-Attention.
"""

from typing import Literal, Optional, Sequence, Tuple
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


class MultiOrderNGramConv(nn.Module):
    """Multi-Order Acoustic N-Gram Convolution.

    Captures speech coarticulation dynamics across multiple continuous acoustic n-gram orders:
    - 1-gram (unigram, K=1): Instantaneous acoustic frame representation (~20 ms)
    - 2-gram (bigram, K=3, pad=1): Immediate diphone transition / coarticulation (~60 ms)
    - 3-gram (trigram, K=5, pad=2): Triphone / syllable nucleus context (~100 ms)
    - 5-gram (5-gram, K=9, pad=4): Sub-word / syllabic span context (~180 ms)

    Each acoustic n-gram branch is processed via depthwise separable convolutions
    and fused via a dynamic softmax gating mechanism.
    """

    def __init__(
        self,
        dim: int,
        ngram_orders: Sequence[int] = (1, 3, 5, 9),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.ngram_orders = tuple(ngram_orders)

        branches = []
        for k in self.ngram_orders:
            if k == 1:
                branches.append(nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
            else:
                branches.append(nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
        self.branches = nn.ModuleList(branches)

        # Dynamic N-gram order gating router
        self.gate_router = nn.Sequential(
            nn.Linear(dim, len(self.ngram_orders)),
            nn.Softmax(dim=-1),
        )
        self.fusion = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: Acoustic features (batch_size, time_frames, dim)

        Returns:
            out: N-gram enriched acoustic features (batch_size, time_frames, dim)
            ngram_weights: Dynamic n-gram order gating weights (batch_size, time_frames, num_orders)
        """
        batch_size, seq_len, dim = x.size()
        x_trans = x.transpose(1, 2)  # (B, D, T) for Conv1d

        branch_outs = []
        for k, branch in zip(self.ngram_orders, self.branches):
            if k == 1:
                branch_outs.append(branch(x))  # (B, T, D)
            else:
                branch_outs.append(branch(x_trans).transpose(1, 2))  # (B, T, D)

        # Stack branch outputs: (B, T, num_orders, D)
        stacked_branches = torch.stack(branch_outs, dim=2)

        # Compute dynamic n-gram gating weights per frame: (B, T, num_orders, 1)
        ngram_weights = self.gate_router(x).unsqueeze(-1)

        # Weighted combination across n-gram orders: (B, T, D)
        fused = (stacked_branches * ngram_weights).sum(dim=2)
        fused = self.fusion(fused)

        out = self.norm(x + torch.tanh(self.gate) * fused)
        return out, ngram_weights.squeeze(-1)


class NGramContextAttention(nn.Module):
    """Local Windowed N-Gram Contextual Self-Attention.

    Operates over temporally downsampled acoustic tokens (~100 ms per token).
    Applies multi-head self-attention with a localized sliding n-gram window mask,
    enforcing local token-level n-gram co-occurrence constraints (e.g. 5-gram token context,
    corresponding to ~500 ms word-level spans in Turkish) without distal interference.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 5,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
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
        self.gate = nn.Parameter(torch.zeros(1))

    def _generate_ngram_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Creates a boolean attention mask where tokens outside the local n-gram window are masked out."""
        indices = torch.arange(seq_len, device=device)
        distance = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()
        half_window = self.window_size // 2
        # Mask is True where attention is NOT allowed
        mask = distance > half_window
        return mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        norm_x = self.norm1(x)

        # Local n-gram window mask: (seq_len, seq_len)
        ngram_mask = self._generate_ngram_window_mask(seq_len, x.device)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        attn_out, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=ngram_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm2(x + torch.tanh(self.gate) * self.dropout(attn_out))
        x = x + self.ffn(x)
        return x


class NGramAttentionProjector(nn.Module):
    """N-Gram Attention Projector for SLAM-ASR (arXiv:2402.08846 + N-Gram extension).

    Architecture Pipeline:
    1. Multi-Order Acoustic N-Gram Convolutions: Gathers multi-scale coarticulation contexts (1, 3, 5, 9).
    2. Temporal Downsampler: k=5 frame concatenation (50 Hz -> 10 Hz).
    3. Base Projection: Maps concatenated features to LLM embedding dimension.
    4. Local Windowed N-Gram Contextual Attention: Models token-level n-gram sequence dependencies.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        ngram_orders: Sequence[int] = (1, 3, 5, 9),
        ngram_window_size: int = 5,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        projector_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
    ):
        super().__init__()
        self.ngram_conv = MultiOrderNGramConv(
            dim=encoder_dim,
            ngram_orders=ngram_orders,
            dropout=attention_dropout,
        )
        self.downsampler = FrameDownsampler(k=downsample_rate, padding_mode=padding_mode)
        concatenated_dim = encoder_dim * downsample_rate

        if projector_type == "linear":
            self.base_projector = LinearProjector(concatenated_dim, llm_dim)
        elif projector_type == "mlp":
            self.base_projector = MLPProjector(concatenated_dim, llm_dim, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported base projector_type: {projector_type}")

        self.ngram_attn = NGramContextAttention(
            dim=llm_dim,
            window_size=ngram_window_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )

    def forward(
        self,
        speech_features: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Multi-order continuous acoustic n-gram contextualization
        ngram_features, _ = self.ngram_conv(speech_features)
        # 2. Temporal downsampling (k=5 concatenation)
        downsampled, mask = self.downsampler(ngram_features, speech_attention_mask)
        # 3. Base projection into LLM space
        projected = self.base_projector(downsampled)
        # 4. Local windowed n-gram token contextual self-attention
        out = self.ngram_attn(projected, mask)
        return out, mask


class SpeechProjector(nn.Module):
    """Unified Speech Projector for SLAM-ASR.

    Supported `projector_type` options:
        - 'linear': Direct concatenation of k frames + Linear layer (SLAM-ASR baseline).
        - 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
        - 'ngram_attention' (or 'ngram', 'ng'): Multi-Order Acoustic N-Gram Convolutions
          + Temporal Downsampling + Local Windowed N-Gram Contextual Self-Attention.

    Args:
        encoder_dim: Dimension of speech encoder output features (e.g. 1280 for Whisper-Large-v3).
        llm_dim: Dimension of LLM input embeddings (e.g. 3584 for Qwen2.5-7B).
        downsample_rate: Downsampling factor k (default: 5).
        projector_type: 'linear', 'mlp', or 'ngram_attention' (default: 'ngram_attention').
        hidden_dim: Hidden dimension when projector_type='mlp' (default: 2048).
        padding_mode: 'discard' or 'pad' for downsampler remainder.
        ngram_orders: Acoustic n-gram kernel sizes (default: (1, 3, 5, 9)).
        ngram_window_size: Window size for token-level n-gram attention (default: 5).
        num_attention_heads: Number of attention heads (default: 4).
        attention_dropout: Dropout probability for attention layers (default: 0.1).
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        projector_type: Literal["linear", "mlp", "ngram_attention", "ngram", "ng"] = "ngram_attention",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
        ngram_orders: Sequence[int] = (1, 3, 5, 9),
        ngram_window_size: int = 5,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.downsample_rate = downsample_rate
        self.projector_type = projector_type

        if projector_type in ("ngram_attention", "ngram", "ng"):
            self.impl = NGramAttentionProjector(
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
                downsample_rate=downsample_rate,
                ngram_orders=ngram_orders,
                ngram_window_size=ngram_window_size,
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
                    f"Choose 'linear', 'mlp', or 'ngram_attention'."
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
