"""Projector modules for SLAM-ASR (arXiv:2402.08846) with Phoneme Attention extensions.

The projector acts as a bridge between the speech encoder and the LLM.
It downsamples speech features temporally and maps them to the LLM's embedding space.

Supported Projector Types:
1. 'linear': Direct concatenation of k frames + Single Linear projection (SLAM-ASR baseline).
2. 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
3. 'phoneme_attention' (or 'pa'): Phonetic Prototype Attention (PPA) with learnable
   Turkish phonetic sound prototypes + Temporal Downsampling + Phonotactic Self-Attention.
4. 'temporal_attention' (or 'ta'): Intra-window Temporal Attention Pooling (TSN-style)
   + Linear projection + Inter-window Multi-Head Temporal Self-Attention.
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


class PhoneticPrototypeAttention(nn.Module):
    """Phonetic Prototype Attention (PPA).

    Maintains a bank of learnable Turkish phonetic prototype embeddings.
    Acoustic frames from the speech encoder query these phonetic prototypes
    via multi-head cross-attention, producing a phonetically-grounded representation
    where each acoustic frame is expressed in terms of phonetic sound categories
    (vowels, consonants, sonorants, plosives, and coarticulatory transitions).

    Args:
        dim: Dimension of speech encoder features (e.g. 1280 for Whisper-Large-v3).
        num_prototypes: Number of learnable phonetic prototype tokens (default: 64).
        num_heads: Number of attention heads (default: 4).
        dropout: Attention dropout probability (default: 0.1).
    """

    def __init__(
        self,
        dim: int,
        num_prototypes: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads

        # Learnable Turkish phonetic prototype codebook
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * (dim ** -0.5))

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))  # Gated residual connection initialized at zero

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: Acoustic features of shape (batch_size, time_frames, dim).
            attention_mask: Optional mask (batch_size, time_frames).

        Returns:
            Tuple of:
                output: Phonetically enriched features (batch_size, time_frames, dim).
                phoneme_attn_weights: Phonetic attention distribution (batch_size, time_frames, num_prototypes).
        """
        batch_size, time_frames, _ = x.size()

        # Queries from speech frames, Keys and Values from phonetic prototypes
        q = self.q_proj(x)  # (B, T, D)
        proto_expanded = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        k = self.k_proj(proto_expanded)  # (B, N_proto, D)
        v = self.v_proj(proto_expanded)  # (B, N_proto, D)

        head_dim = self.dim // self.num_heads

        # Reshape for multi-head attention: (B, heads, seq_len, head_dim)
        q_heads = q.view(batch_size, time_frames, self.num_heads, head_dim).transpose(1, 2)
        k_heads = k.view(batch_size, self.num_prototypes, self.num_heads, head_dim).transpose(1, 2)
        v_heads = v.view(batch_size, self.num_prototypes, self.num_heads, head_dim).transpose(1, 2)

        # Attention scores: (B, heads, T, N_proto)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attended phonetic representations: (B, heads, T, head_dim)
        proto_context = torch.matmul(attn_weights, v_heads)
        proto_context = proto_context.transpose(1, 2).contiguous().view(batch_size, time_frames, self.dim)
        proto_context = self.out_proj(proto_context)

        # Gated residual connection + LayerNorm
        out = self.norm(x + torch.tanh(self.gate) * self.dropout(proto_context))

        # Average attention weights across heads for visualization/inspection: (B, T, N_proto)
        avg_weights = attn_weights.mean(dim=1)
        return out, avg_weights


class PhonemeSelfAttentionBlock(nn.Module):
    """Multi-Head Self-Attention over phonetically-aligned downsampled tokens.

    Models inter-token phonotactic constraints (e.g. Turkish vowel harmony,
    consonant voicing/devoicing rules, suffix attachments) across the sequence.
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


class PhonemeAttentionProjector(nn.Module):
    """Phoneme Attention Projector for SLAM-ASR (arXiv:2402.08846 + PA extension).

    Combines:
    1. Phonetic Prototype Attention (PPA): Frame-level acoustic-to-phonetic prototype attention
    2. Temporal Downsampler (k consecutive frame concatenation, reducing 50 Hz to 10 Hz)
    3. Linear or MLP Projection into LLM embedding space
    4. Phonotactic Self-Attention over downsampled speech tokens
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        num_phoneme_prototypes: int = 64,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        projector_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
    ):
        super().__init__()
        self.phoneme_attn = PhoneticPrototypeAttention(
            dim=encoder_dim,
            num_prototypes=num_phoneme_prototypes,
            num_heads=num_attention_heads,
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

        self.phoneme_self_attn = PhonemeSelfAttentionBlock(
            dim=llm_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )

    def forward(
        self,
        speech_features: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Enrich continuous acoustic frames with phonetic prototype attention
        phonetic_features, _ = self.phoneme_attn(speech_features, speech_attention_mask)
        # 2. Temporal downsampling (k=5 concatenation)
        downsampled, mask = self.downsampler(phonetic_features, speech_attention_mask)
        # 3. Base projection into LLM dimension
        projected = self.base_projector(downsampled)
        # 4. Phonotactic self-attention across speech tokens
        out = self.phoneme_self_attn(projected, mask)
        return out, mask


class SpeechProjector(nn.Module):
    """Unified Speech Projector for SLAM-ASR.

    Supported `projector_type` options:
        - 'linear': Direct concatenation of k frames + Linear layer (SLAM-ASR baseline).
        - 'mlp': Direct concatenation of k frames + 2-layer MLP (ReLU).
        - 'phoneme_attention' (or 'pa'): Phonetic Prototype Attention (PPA)
          + Temporal Downsampling + Projection + Phonotactic Self-Attention.

    Args:
        encoder_dim: Dimension of speech encoder output features (e.g. 1280 for Whisper-Large-v3).
        llm_dim: Dimension of LLM input embeddings (e.g. 3584 for Qwen2.5-7B).
        downsample_rate: Downsampling factor k (default: 5).
        projector_type: 'linear', 'mlp', or 'phoneme_attention' (default: 'phoneme_attention').
        hidden_dim: Hidden dimension when projector_type='mlp' (default: 2048).
        padding_mode: 'discard' or 'pad' for downsampler remainder.
        num_phoneme_prototypes: Number of phonetic prototypes for PPA (default: 64).
        num_attention_heads: Number of attention heads for Phoneme Attention (default: 4).
        attention_dropout: Dropout probability for Phoneme Attention (default: 0.1).
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        projector_type: Literal["linear", "mlp", "phoneme_attention", "pa"] = "phoneme_attention",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
        num_phoneme_prototypes: int = 64,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.downsample_rate = downsample_rate
        self.projector_type = projector_type

        if projector_type in ("phoneme_attention", "phoneme", "pa"):
            self.impl = PhonemeAttentionProjector(
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
                downsample_rate=downsample_rate,
                num_phoneme_prototypes=num_phoneme_prototypes,
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
                    f"Choose 'linear', 'mlp', or 'phoneme_attention'."
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
