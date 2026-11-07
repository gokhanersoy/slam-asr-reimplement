"""Speech encoder wrapper for SLAM-ASR (arXiv:2402.08846).

Supports:
- WavLM (e.g. microsoft/wavlm-large, microsoft/wavlm-base-plus)
- HuBERT (e.g. facebook/hubert-large-ls960-ft, facebook/hubert-xlarge-ls960-ft)
- Wav2Vec2 (e.g. facebook/wav2vec2-base-960h)
- Whisper Encoder (e.g. openai/whisper-large-v2, openai/whisper-small)
"""

from typing import Optional, Union
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoFeatureExtractor,
    WhisperModel,
    WhisperForConditionalGeneration,
)


class SpeechEncoderWrapper(nn.Module):
    """Unified wrapper around pre-trained speech encoders used in SLAM-ASR.

    Extracts frame-level representations at ~50 Hz from raw waveform or mel spectrograms.
    In the paper, the speech encoder is kept FROZEN to prevent overfitting on limited ASR data
    and to preserve the foundational acoustic representations.

    Args:
        encoder_name_or_path: Hugging Face model ID or local directory.
        freeze_encoder: If True, keep all encoder weights frozen (default: True, per SLAM-ASR).
        model_type: Optional explicit model family ('hubert', 'wavlm', 'whisper', 'auto').
    """

    def __init__(
        self,
        encoder_name_or_path: str,
        freeze_encoder: bool = True,
        model_type: str = "auto",
    ):
        super().__init__()
        self.encoder_name_or_path = encoder_name_or_path
        self.freeze_encoder = freeze_encoder

        lowered = encoder_name_or_path.lower()
        if model_type == "auto":
            if "whisper" in lowered:
                self.encoder_type = "whisper"
            elif "wavlm" in lowered:
                self.encoder_type = "wavlm"
            elif "hubert" in lowered:
                self.encoder_type = "hubert"
            else:
                self.encoder_type = "wav2vec2"
        else:
            self.encoder_type = model_type.lower()

        # Load underlying backbone
        if self.encoder_type == "whisper":
            try:
                whisper_full = WhisperForConditionalGeneration.from_pretrained(encoder_name_or_path)
                self.encoder = whisper_full.get_encoder()
            except Exception:
                whisper_full = WhisperModel.from_pretrained(encoder_name_or_path)
                self.encoder = whisper_full.encoder
            self.hidden_size = self.encoder.config.d_model
        else:
            # HuBERT, WavLM, Wav2Vec2
            self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
            self.hidden_size = self.encoder.config.hidden_size

        if self.freeze_encoder:
            self.freeze()

    def freeze(self):
        """Freeze all parameters in the speech encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze(self):
        """Unfreeze speech encoder for fine-tuning exploration."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract acoustic representations.

        Args:
            input_values:
                - For raw audio models (HuBERT/WavLM): raw waveforms of shape (batch, time_samples)
                - For Whisper: log-mel spectrogram features of shape (batch, n_mels, time_frames)
            attention_mask: Optional mask for variable-length inputs (batch, time)

        Returns:
            Tuple of:
                hidden_states: (batch, time_frames, hidden_size) at ~50 Hz.
                feature_attention_mask: Updated attention mask corresponding to output time frames.
        """
        # If encoder is frozen, compute under torch.no_grad() when model is in train mode
        context = torch.no_grad() if (self.freeze_encoder and not self.training) else torch.enable_grad()

        with context:
            if self.encoder_type == "whisper":
                # input_values: (batch, 80/128, 3000)
                outputs = self.encoder(input_values)
                hidden_states = outputs.last_hidden_state
                # Whisper output length is fixed at 1500 frames for 30s input
                output_mask = None
                if attention_mask is not None:
                    # Whisper uses 2 conv layers with stride 2, subsampling rate = 2
                    output_len = hidden_states.size(1)
                    output_mask = attention_mask[:, ::2][:, :output_len]
                return hidden_states, output_mask

            else:
                # HuBERT, WavLM, Wav2Vec2 take raw waveform: (batch, samples)
                kwargs = {}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask

                outputs = self.encoder(input_values, **kwargs)
                hidden_states = outputs.last_hidden_state

                output_mask = None
                if attention_mask is not None:
                    # Calculate output length through feature extractor conv layers
                    if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
                        lengths = attention_mask.sum(dim=-1)
                        out_lengths = self.encoder._get_feat_extract_output_lengths(lengths)
                        max_len = hidden_states.size(1)
                        batch_size = hidden_states.size(0)
                        output_mask = torch.arange(max_len, device=attention_mask.device).unsqueeze(0).expand(batch_size, -1) < out_lengths.unsqueeze(1)
                        output_mask = output_mask.long()

                return hidden_states, output_mask
