"""Enhanced Projector modules for SLAM-ASR (t1-enhanced).

Upgrades the Baseline (t0) with:
1. FrameDownsampler: Temporal pooling by grouping k consecutive frames (k=4, 5, 6).
2. Conv1dDownsampler: 1D Depthwise-Separable Convolutional downsampling with stride k.
3. LinearProjector: Single linear projection layer.
4. EnhancedMLPProjector: Two-layer MLP with modern non-linear activations
   (GELU, SwiGLU, or ReLU), LayerNorm, and Dropout.
5. SpeechProjector: Unified interface orchestrating all projector options.
"""

from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameDownsampler(nn.Module):
    """Downsamples speech features along the temporal dimension by concatenating
    every k consecutive frames in the feature dimension.
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
        downsampled = x.contiguous().view(batch_size, new_seq_len, dim * self.k)

        downsampled_mask = None
        if attention_mask is not None:
            downsampled_mask = (
                attention_mask.view(batch_size, new_seq_len, self.k).sum(dim=-1) > 0
            ).long()

        return downsampled, downsampled_mask


class Conv1dDownsampler(nn.Module):
    """1D Depthwise-Separable Convolutional Downsampler.

    Downsamples speech representations along time from T to T//k using a stride of k,
    capturing continuous local acoustic context via convolution instead of simple concatenation.
    """

    def __init__(self, in_dim: int, out_dim: Optional[int] = None, k: int = 5):
        super().__init__()
        self.k = k
        out_dim = out_dim or in_dim
        kernel_size = 2 * k - 1
        padding = k - 1
        self.depthwise = nn.Conv1d(
            in_dim, in_dim, kernel_size=kernel_size, stride=k, padding=padding, groups=in_dim
        )
        self.pointwise = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, T, D) -> (B, D, T)
        x_trans = x.transpose(1, 2)
        out = self.depthwise(x_trans)
        out = self.pointwise(out)
        out = out.transpose(1, 2)  # (B, T//k, out_dim)
        out = self.norm(self.act(out))

        downsampled_mask = None
        if attention_mask is not None:
            new_len = out.size(1)
            downsampled_mask = attention_mask[:, ::self.k][:, :new_len]
        return out, downsampled_mask


class LinearProjector(nn.Module):
    """Single linear projector mapping downsampled features to LLM embedding dimension."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class EnhancedMLPProjector(nn.Module):
    """Two-layer non-linear MLP projector for SLAM-ASR with modern activations
    (GELU, SwiGLU, or ReLU), LayerNorm, and Dropout.

    Structure:
      Linear(in_dim, hidden_dim) -> LayerNorm -> Activation -> Dropout -> Linear(hidden_dim, out_dim) -> LayerNorm
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.activation_name = activation.lower()
        if self.activation_name == "swiglu":
            self.linear_gate = nn.Linear(in_dim, hidden_dim)
            self.linear_up = nn.Linear(in_dim, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.linear1 = nn.Linear(in_dim, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
            if self.activation_name == "relu":
                self.act = nn.ReLU()
            else:
                self.act = nn.GELU()
            self.linear2 = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "swiglu":
            gate = F.silu(self.linear_gate(x))
            up = self.linear_up(x)
            hidden = self.norm1(gate * up)
            hidden = self.dropout(hidden)
            out = self.linear2(hidden)
        else:
            hidden = self.linear1(x)
            hidden = self.norm1(hidden)
            hidden = self.dropout(self.act(hidden))
            out = self.linear2(hidden)
        return self.norm_out(out)


class MLPProjector(nn.Module):
    """Classic two-layer MLP projector matching paper Section 2.3 & 3.1.3:
    Linear(in_dim, hidden_dim) -> ReLU() -> Linear(hidden_dim, out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class SpeechProjector(nn.Module):
    """Unified Speech Projector for SLAM-ASR (t1-enhanced).

    Supported projector_type:
      - 'mlp': 2-layer MLP with non-linear activation (GELU/SwiGLU/ReLU) + k-frame downsampling
      - 'conv_mlp': 1D depthwise-separable CNN Downsampling + 2-layer MLP
      - 'swiglu': 2-layer SwiGLU MLP projector
      - 'linear': classic frame concatenation + single linear layer (Baseline t0)
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        projector_type: str = "mlp",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
        mlp_activation: str = "gelu",
        mlp_dropout: float = 0.1,
        downsampler_type: str = "concat",
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.downsample_rate = downsample_rate
        self.projector_type = projector_type.lower()
        self.downsampler_type = downsampler_type.lower()

        if self.downsampler_type == "conv1d" or self.projector_type == "conv_mlp":
            self.downsampler = Conv1dDownsampler(
                in_dim=encoder_dim,
                out_dim=encoder_dim,
                k=downsample_rate,
            )
            projector_in_dim = encoder_dim
        else:
            self.downsampler = FrameDownsampler(
                k=downsample_rate,
                padding_mode=padding_mode,
            )
            projector_in_dim = encoder_dim * downsample_rate

        if self.projector_type in ("mlp", "conv_mlp"):
            self.projector = EnhancedMLPProjector(
                in_dim=projector_in_dim,
                out_dim=llm_dim,
                hidden_dim=hidden_dim,
                activation=mlp_activation,
                dropout=mlp_dropout,
            )
        elif self.projector_type == "swiglu":
            self.projector = EnhancedMLPProjector(
                in_dim=projector_in_dim,
                out_dim=llm_dim,
                hidden_dim=hidden_dim,
                activation="swiglu",
                dropout=mlp_dropout,
            )
        elif self.projector_type == "linear":
            self.projector = LinearProjector(
                in_dim=projector_in_dim,
                out_dim=llm_dim,
            )
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")

    def forward(
        self,
        speech_features: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        downsampled, mask = self.downsampler(speech_features, speech_attention_mask)
        projected = self.projector(downsampled)
        return projected, mask
