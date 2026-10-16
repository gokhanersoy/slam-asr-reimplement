"""Projector modules for SLAM-ASR (arXiv:2402.08846).

The projector acts as a bridge between the speech encoder and the LLM.
It downsamples speech features temporally by concatenating k consecutive frames
in the feature dimension, and maps them to the LLM's embedding space using
a linear layer or a 2-layer MLP with ReLU activation.
"""

from typing import Literal, Optional
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

    Maps downsampled speech features (dim * k) directly to LLM hidden dimension.
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


class SpeechProjector(nn.Module):
    """Unified Speech Projector for SLAM-ASR.

    Combines temporal downsampling (k consecutive frame concatenation)
    with projection (linear or MLP) into the LLM embedding space.

    Args:
        encoder_dim: Dimension of speech encoder output features (e.g., 1024 for WavLM/HuBERT-large).
        llm_dim: Dimension of LLM input embeddings (e.g., 4096 for LLaMA/Vicuna-7B, 2048 for TinyLlama).
        downsample_rate: Downsampling factor k (default: 5).
        projector_type: 'linear' or 'mlp' (default: 'linear' as in final SLAM-ASR benchmark).
        hidden_dim: Hidden dimension when projector_type='mlp' (default: 2048).
        padding_mode: 'discard' or 'pad' for downsampler remainder.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 5,
        projector_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 2048,
        padding_mode: Literal["discard", "pad"] = "discard",
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.downsample_rate = downsample_rate
        self.projector_type = projector_type
        self.downsampler = FrameDownsampler(k=downsample_rate, padding_mode=padding_mode)

        concatenated_dim = encoder_dim * downsample_rate
        if projector_type == "linear":
            self.projector = LinearProjector(concatenated_dim, llm_dim)
        elif projector_type == "mlp":
            self.projector = MLPProjector(concatenated_dim, llm_dim, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}. Choose 'linear' or 'mlp'.")

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
        downsampled, mask = self.downsampler(speech_features, speech_attention_mask)
        projected = self.projector(downsampled)
        return projected, mask
