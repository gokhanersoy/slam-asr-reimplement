"""Batch collator for SLAM-ASR."""

from typing import Any, Dict, List
import torch
import torch.nn.functional as F


class ASRDataCollator:
    """Collates a list of ASRDataset items into a batched dictionary

    ready for model forward pass.

    Handles variable-length raw audio waveforms or fixed-size mel spectrograms.
    """

    def __init__(self, is_whisper: bool = False):
        self.is_whisper = is_whisper

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = [item["key"] for item in batch]
        targets = [item["target"] for item in batch]
        prompts = [item["prompt"] for item in batch]

        speeches = [item["speech"] for item in batch]

        if self.is_whisper:
            # Speeches are (n_mels, 3000), stack along batch dimension
            speech_inputs = torch.stack(speeches, dim=0)  # (batch, n_mels, 3000)
            speech_attention_mask = None
        else:
            # Speeches are 1D variable length waveforms (samples,)
            lengths = [s.size(0) for s in speeches]
            max_len = max(lengths)
            batch_size = len(speeches)

            speech_inputs = torch.zeros(batch_size, max_len, dtype=torch.float32)
            speech_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

            for i, s in enumerate(speeches):
                cur_len = s.size(0)
                speech_inputs[i, :cur_len] = s
                speech_attention_mask[i, :cur_len] = 1

        return {
            "keys": keys,
            "speech_inputs": speech_inputs,
            "speech_attention_mask": speech_attention_mask,
            "transcripts": targets,
            "prompts": prompts,
        }
