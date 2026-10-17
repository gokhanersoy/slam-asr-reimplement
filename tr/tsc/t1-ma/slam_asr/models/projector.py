"""Projector modules for SLAM-ASR (arXiv:2402.08846) with Morpheme Attention extensions.

The projector acts as a bridge between the speech encoder and the LLM.
It downsamples speech features temporally and maps them to the LLM's embedding space.

Supported Projector Types:
1. 'linear': Direct concatenation of k frames + Single Linear projection (SLAM-ASR baseline).
2. 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
3. 'morpheme_attention' (or 'ma'): Multi-Scale Morphemic Convolutions + Temporal Downsampling
   + Morpheme Prototype Cross-Attention (128 morphological category slots)
   + Inter-Morphemic Contextual Self-Attention.
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


class MultiScaleMorphemeConv(nn.Module):
    """Multi-Scale Depthwise Convolution for capturing variable-length morpheme spans.

    In Turkish, roots and suffixes span varying temporal durations:
        - Short affixes (e.g. -i, -e, -m): ~60 ms (kernel 3)
        - Standard suffixes (e.g. -ler, -den, -miş): ~100 ms (kernel 5)
        - Roots and compound morphemes (e.g. kitap-, gel-, -ebil-): ~220 ms (kernel 11)

    Processes acoustic features across parallel depthwise 1D convolutional branches
    and fuses them with a learned channel gating mechanism.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tuple[int, ...] = (3, 5, 11),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])
        self.fusion = nn.Linear(dim * len(kernel_sizes), dim)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> transpose to (B, D, T) for Conv1d
        x_conv = x.transpose(1, 2)
        branch_outs = [b(x_conv).transpose(1, 2) for b in self.branches]
        concat = torch.cat(branch_outs, dim=-1)
        fused = self.fusion(concat)
        out = self.norm(x + torch.tanh(self.gate) * fused)
        return out


class MorphemePrototypeAttention(nn.Module):
    """Morpheme Prototype Cross-Attention (MPCA).

    Maintains a bank of learnable Turkish morphological prototypes (roots, case suffixes,
    possessive markers, tense/aspect affixes, person endings, derivational suffixes).
    Downsampled speech tokens query these morphological prototype slots via multi-head
    cross-attention, mapping acoustic representations into functional morphological roles.
    """

    def __init__(
        self,
        dim: int,
        num_prototypes: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads

        # Learnable morphological functional prototype slots
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * (dim ** -0.5))

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        proto_expanded = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        k = self.k_proj(proto_expanded)
        v = self.v_proj(proto_expanded)

        head_dim = self.dim // self.num_heads

        q_heads = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k_heads = k.view(batch_size, self.num_prototypes, self.num_heads, head_dim).transpose(1, 2)
        v_heads = v.view(batch_size, self.num_prototypes, self.num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_probs = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_probs)

        context = torch.matmul(attn_weights, v_heads)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        context = self.out_proj(context)

        out = self.norm(x + torch.tanh(self.gate) * self.dropout(context))
        avg_weights = attn_probs.mean(dim=1)
        return out, avg_weights


class MorphemeContextSelfAttention(nn.Module):
    """Inter-Morphemic Contextual Self-Attention.

    Models grammatical dependencies across the sequence (e.g. root-suffix vowel harmony,
    subject-verb agreement, and sequential affix ordering constraints).
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
        norm_x = self.norm1(x)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        attn_out, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class MorphemeAttentionProjector(nn.Module):
    """Morpheme Attention Projector for SLAM-ASR (arXiv:2402.08846 + MA extension).

    Combines:
    1. Multi-Scale Morphemic Convolutions: Gathers multi-scale acoustic context for varying affix durations.
    2. Temporal Downsampler (k=5 concatenation, 50 Hz -> 10 Hz).
    3. Base Projection to LLM embedding dimension.
    4. Morpheme Prototype Cross-Attention (MPCA): Interacts with 128 morphological category prototypes.
    5. Inter-Morphemic Contextual Self-Attention: Enforces vowel harmony and grammatical coherence.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        num_morpheme_prototypes: int = 128,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        projector_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
    ):
        super().__init__()
        self.multiscale_conv = MultiScaleMorphemeConv(
            dim=encoder_dim,
            kernel_sizes=(3, 5, 11),
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

        self.morpheme_proto_attn = MorphemePrototypeAttention(
            dim=llm_dim,
            num_prototypes=num_morpheme_prototypes,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )
        self.morpheme_self_attn = MorphemeContextSelfAttention(
            dim=llm_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )

    def forward(
        self,
        speech_features: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Multi-scale morphemic contextualization (spans of 60ms, 100ms, 220ms)
        conv_features = self.multiscale_conv(speech_features)
        # 2. Temporal downsampling (k=5 concatenation)
        downsampled, mask = self.downsampler(conv_features, speech_attention_mask)
        # 3. Base projection into LLM space
        projected = self.base_projector(downsampled)
        # 4. Morpheme Prototype Cross-Attention (128 morphological slots)
        morpheme_features, _ = self.morpheme_proto_attn(projected, mask)
        # 5. Inter-morphemic contextual self-attention (vowel harmony & agreement)
        out = self.morpheme_self_attn(morpheme_features, mask)
        return out, mask


class SpeechProjector(nn.Module):
    """Unified Speech Projector for SLAM-ASR.

    Supported `projector_type` options:
        - 'linear': Direct concatenation of k frames + Linear layer (SLAM-ASR baseline).
        - 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
        - 'morpheme_attention' (or 'ma'): Multi-Scale Morphemic Convolutions
          + Temporal Downsampling + Morpheme Prototype Attention + Inter-Morphemic Self-Attention.

    Args:
        encoder_dim: Dimension of speech encoder output features (e.g. 1280 for Whisper-Large-v3).
        llm_dim: Dimension of LLM input embeddings (e.g. 3584 for Qwen2.5-7B).
        downsample_rate: Downsampling factor k (default: 5).
        projector_type: 'linear', 'mlp', or 'morpheme_attention' (default: 'morpheme_attention').
        hidden_dim: Hidden dimension when projector_type='mlp' (default: 2048).
        padding_mode: 'discard' or 'pad' for downsampler remainder.
        num_morpheme_prototypes: Number of morphological prototypes (default: 128).
        num_attention_heads: Number of attention heads (default: 4).
        attention_dropout: Dropout probability for attention layers (default: 0.1).
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        projector_type: Literal["linear", "mlp", "morpheme_attention", "ma"] = "morpheme_attention",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
        num_morpheme_prototypes: int = 128,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.downsample_rate = downsample_rate
        self.projector_type = projector_type

        if projector_type in ("morpheme_attention", "morpheme", "ma"):
            self.impl = MorphemeAttentionProjector(
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
                downsample_rate=downsample_rate,
                num_morpheme_prototypes=num_morpheme_prototypes,
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
                    f"Choose 'linear', 'mlp', or 'morpheme_attention'."
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
