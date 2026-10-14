"""Dataset implementations for SLAM-ASR.

Supports:
- JSONL files formatted as: {"key": "...", "source": "path/to/audio.wav", "target": "transcript"}
- Hugging Face datasets (e.g. "librispeech_asr", splits like "train.clean.100", "validation.clean", "test.other")
- In-memory lists of audio-text pairs
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

from slam_asr.data.audio_processor import AudioProcessor


class ASRDataset(Dataset):
    """PyTorch Dataset for SLAM-ASR training and evaluation.

    Args:
        data_path_or_dataset: Path to a JSONL file, a directory with manifest,
            or an already loaded list of dict items / Hugging Face dataset.
        audio_processor: AudioProcessor instance for loading and featurizing audio.
        prompt: Optional default prompt for each sample.
    """

    def __init__(
        self,
        data_source: Union[str, List[Dict[str, Any]], Any],
        audio_processor: Optional[AudioProcessor] = None,
        prompt: str = "Transcribe speech to text.",
    ):
        super().__init__()
        self.audio_processor = audio_processor or AudioProcessor()
        self.default_prompt = prompt
        self.examples: List[Dict[str, Any]] = []

        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Data file not found: {data_source}")

            # Parse JSONL or manifest file
            with open(data_source, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        self.examples.append(item)
        elif isinstance(data_source, list):
            self.examples = data_source
        else:
            # Assume Hugging Face Dataset-like object
            self.examples = list(data_source)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]

        # Extract audio source
        audio_source = item.get("source") or item.get("audio") or item.get("file")
        if isinstance(audio_source, dict) and "array" in audio_source:
            # Hugging Face Audio feature dict
            audio_tensor = self.audio_processor.process(audio_source["array"])
        else:
            audio_tensor = self.audio_processor.process(audio_source)

        # Extract target text
        target_text = (
            item.get("target")
            or item.get("text")
            or item.get("transcript")
            or item.get("sentence")
            or ""
        )

        prompt = item.get("prompt", self.default_prompt)
        key = item.get("key", str(idx))

        return {
            "key": key,
            "speech": audio_tensor,
            "target": target_text,
            "prompt": prompt,
        }
