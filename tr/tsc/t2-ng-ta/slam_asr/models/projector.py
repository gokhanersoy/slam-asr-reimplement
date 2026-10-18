"""SLAM-ASR Speech Projector Modules with Hierarchical NGram-Temporal Hybrid Attention (t2-ng-ta).

This module implements the unified speech projector architectures for SLAM-ASR:
1. FrameDownsampler: Temporal pooling by grouping k consecutive frames.
2. LinearProjector: Single linear layer mapping downsampled speech to LLM embedding dimension.
3. MLPProjector: Two-layer MLP with ReLU activation as described in Section 2.3 & 3.1.3.
4. MultiOrderNGramConv: Multi-order acoustic n-gram convolutions with dynamic softmax gating.
5. TemporalAttentionPooling: Intra-window soft attention pooling.
6. TemporalAttentionDownsampler: Segmental downsampler combining window grouping and attention centroid.
7. NGramContextAttention: Local windowed sliding attention over acoustic tokens.
8. TemporalSelfAttentionBlock: Global multi-head self-attention over full utterance.
9. HybridTemporalNGramAttentionBlock: Dual-scale local-global attention block (split-heads or sequential).
10. NGramTemporalHybridProjector: End-to-end hybrid architecture combining Stage 1 through Stage 4.
11. SpeechProjector: Unified interface orchestrating all projector variants.
"""

from typing import Literal, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameDownsampler(nn.Module):
    """Downsamples speech encoder representations along the time dimension by grouping
    consecutive frames, matching the method described in Section 2.3 & 3.1.3:
    "Speech features are concatenated every k frames, downsampling the frame length
    from T to T/k and expanding the dimension from d to k*d."
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
            else:
                raise ValueError(f"Unknown padding_mode: {self.padding_mode}")

        new_seq_len = x.size(1) // self.k
        x_downsampled = x.contiguous().view(batch_size, new_seq_len, dim * self.k)

        downsampled_mask = None
        if attention_mask is not None:
            downsampled_mask = (
                attention_mask.view(batch_size, new_seq_len, self.k).sum(dim=-1) > 0
            ).long()

        return x_downsampled, downsampled_mask


class LinearProjector(nn.Module):
    """Single linear projector for SLAM-ASR as described in Section 2.3 & 3.1.3:
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

    Captures continuous acoustic coarticulation dynamics across multiple continuous n-gram orders:
    - 1-gram (unigram, K=1): Instantaneous acoustic frame representation (~20 ms)
    - 2-gram (bigram, K=3, pad=1): Immediate diphone transition / coarticulation (~60 ms)
    - 3-gram (trigram, K=5, pad=2): Triphone / syllable nucleus context (~100 ms)
    - 5-gram (5-gram, K=9, pad=4): Sub-word / syllabic span context (~180 ms)

    Each acoustic n-gram branch is processed via depthwise separable convolutions
    and fused via a dynamic softmax gating router.
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


class TemporalAttentionPooling(nn.Module):
    """Segment-level / Intra-window Temporal Attention Pooling.

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


class NGramContextAttention(nn.Module):
    """Local Windowed N-Gram Contextual Self-Attention.

    Operates over temporally downsampled acoustic tokens (~100 ms per token).
    Applies multi-head self-attention with a localized sliding n-gram window mask,
    enforcing local token-level n-gram co-occurrence constraints without distal interference.
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
        """Boolean attention mask: True where attention is DISALLOWED."""
        indices = torch.arange(seq_len, device=device)
        distance = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()
        half_window = self.window_size // 2
        return distance > half_window

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        norm_x = self.norm1(x)

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


class TemporalSelfAttentionBlock(nn.Module):
    """Multi-Head Temporal Self-Attention over downsampled speech representations.

    Allows downsampled acoustic tokens to attend globally across the entire utterance,
    modeling long-range phonotactic patterns, syntax, and morphological vowel harmony.
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
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)
        norm_x = self.norm2(x)
        x = x + self.ffn(norm_x)
        return x


class HybridTemporalNGramAttentionBlock(nn.Module):
    """Hierarchical Hybrid Attention Block combining Local N-Gram Windowed Attention
    and Global Full-Utterance Temporal Self-Attention.

    Modes:
      - 'split_heads' (default): Multi-Head Attention where 50% of the attention heads
        are constrained by a local sliding n-gram window mask (capturing immediate sub-word
        phonetic and morphological transitions, ~500 ms), while the remaining 50% of heads
        are unmasked (attending across the entire utterance to model vowel harmony, syntax,
        and global prosody).
      - 'sequential': Local N-Gram Context Attention layer followed by a Global Temporal
        Self-Attention layer.
      - 'global': Full-utterance multi-head self-attention.
      - 'local': Strict local windowed self-attention.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 5,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        mode: Literal["split_heads", "sequential", "global", "local"] = "split_heads",
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.mode = mode
        ffn_dim = ffn_dim or (dim * 2)

        if mode == "sequential":
            self.local_attn = NGramContextAttention(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            self.global_attn = TemporalSelfAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
        else:
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
        """Boolean attention mask: True where attention is DISALLOWED."""
        indices = torch.arange(seq_len, device=device)
        distance = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()
        half_window = self.window_size // 2
        return distance > half_window

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "sequential":
            x = self.local_attn(x, attention_mask)
            x = self.global_attn(x, attention_mask)
            return x

        batch_size, seq_len, _ = x.size()
        norm_x = self.norm1(x)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        attn_mask = None
        if self.mode == "local":
            attn_mask = self._generate_ngram_window_mask(seq_len, x.device)
        elif self.mode == "split_heads":
            local_mask = self._generate_ngram_window_mask(seq_len, x.device)
            global_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=x.device)
            half_heads = max(self.num_heads // 2, 1)
            head_masks = [
                local_mask if h < half_heads else global_mask
                for h in range(self.num_heads)
            ]
            stacked_masks = torch.stack(head_masks, dim=0)  # (num_heads, seq_len, seq_len)
            attn_mask = stacked_masks.repeat(batch_size, 1, 1)  # (B * num_heads, seq_len, seq_len)
        elif self.mode == "global":
            attn_mask = None

        attn_out, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm2(x + torch.tanh(self.gate) * self.dropout(attn_out))
        x = x + self.ffn(x)
        return x


class NGramTemporalHybridProjector(nn.Module):
    """Hierarchical NGram-Temporal Hybrid Speech Projector (t2-ng-ta).

    Architecture:
    Stage 1: Multi-Order Acoustic N-Gram Conv (Multi-scale acoustic coarticulation, ~20ms frames)
    Stage 2: Segmental Temporal Attention Downsampling (Intra-window attention pooling, k=5)
    Stage 3: Base Projection (Linear or MLP into LLM dimension)
    Stage 4: Hybrid Temporal-NGram Attention Block (Dual-scale local-global token attention)
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        ngram_orders: Sequence[int] = (1, 3, 5, 9),
        ngram_window_size: int = 5,
        temporal_pooling_mode: Literal["hybrid", "attention", "weighted_concat"] = "hybrid",
        hybrid_attention_mode: Literal["split_heads", "sequential", "global", "local"] = "split_heads",
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
        self.downsampler = TemporalAttentionDownsampler(
            k=downsample_rate,
            encoder_dim=encoder_dim,
            pooling_mode=temporal_pooling_mode,
            padding_mode=padding_mode,
        )
        downsampled_dim = self.downsampler.out_dim

        if projector_type == "linear":
            self.base_projector = LinearProjector(downsampled_dim, llm_dim)
        elif projector_type == "mlp":
            self.base_projector = MLPProjector(downsampled_dim, llm_dim, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported base projector_type: {projector_type}")

        self.hybrid_attn = HybridTemporalNGramAttentionBlock(
            dim=llm_dim,
            window_size=ngram_window_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            mode=hybrid_attention_mode,
        )

    def forward(
        self,
        speech_features: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Stage 1: Continuous acoustic n-gram coarticulation
        acoustic_ngram, _ = self.ngram_conv(speech_features)

        # Stage 2: Segmental temporal attention downsampling
        downsampled, mask = self.downsampler(acoustic_ngram, speech_attention_mask)

        # Stage 3: Project to LLM space
        projected = self.base_projector(downsampled)

        # Stage 4: Dual local-global contextual attention
        out = self.hybrid_attn(projected, mask)
        return out, mask


class SpeechProjector(nn.Module):
    """Unified Speech Projector for SLAM-ASR.

    Supported projector_type:
      - 'ngram_temporal_hybrid', 'hybrid', 'ng_ta': Hierarchical NGram-Temporal Hybrid projector (t2-ng-ta)
      - 'ngram_attention': Multi-Order N-Gram Acoustic & Context Attention (t1-ng)
      - 'temporal_attention': Segmental Temporal Attention & Self-Attention (t1-ta)
      - 'linear': Standard frame concatenation + single Linear layer (SLAM-ASR Baseline t0)
      - 'mlp': Standard frame concatenation + 2-layer MLP with ReLU
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        projector_type: str = "ngram_temporal_hybrid",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
        ngram_orders: Sequence[int] = (1, 3, 5, 9),
        ngram_window_size: int = 5,
        temporal_pooling_mode: Literal["hybrid", "attention", "weighted_concat"] = "hybrid",
        hybrid_attention_mode: Literal["split_heads", "sequential", "global", "local"] = "split_heads",
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.downsample_rate = downsample_rate
        self.projector_type = projector_type
        self.padding_mode = padding_mode

        if projector_type in ("ngram_temporal_hybrid", "hybrid", "ng_ta"):
            self.projector = NGramTemporalHybridProjector(
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
                downsample_rate=downsample_rate,
                ngram_orders=ngram_orders,
                ngram_window_size=ngram_window_size,
                temporal_pooling_mode=temporal_pooling_mode,
                hybrid_attention_mode=hybrid_attention_mode,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                projector_type="linear",
                hidden_dim=hidden_dim,
                padding_mode=padding_mode,
            )
            self.downsampler = None
        elif projector_type in ("linear", "mlp"):
            self.downsampler = FrameDownsampler(k=downsample_rate, padding_mode=padding_mode)
            in_dim = encoder_dim * downsample_rate
            if projector_type == "linear":
                self.projector = LinearProjector(in_dim, llm_dim)
            else:
                self.projector = MLPProjector(in_dim, llm_dim, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")

    def forward(
        self,
        speech_features: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.downsampler is not None:
            downsampled, mask = self.downsampler(speech_features, speech_attention_mask)
            projected = self.projector(downsampled)
            return projected, mask
        else:
            return self.projector(speech_features, speech_attention_mask)
