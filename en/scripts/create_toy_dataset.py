#!/usr/bin/env python3
"""Generates synthetic toy audio files and JSONL manifests to test the training and evaluation pipeline."""

import json
import math
import os
import struct
import wave


def generate_wav(file_path: str, duration_sec: float = 2.0, freq: float = 440.0, sr: int = 16000):
    """Write a synthetic sine wave PCM 16-bit mono WAV file using Python standard library."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    num_samples = int(duration_sec * sr)
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(num_samples):
            value = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * freq * (i / sr)))
            wf.writeframes(struct.pack("<h", value))


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = os.path.join(base_dir, "data", "toy_audio")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(audio_dir, exist_ok=True)

    samples = [
        ("toy_1", 2.0, 440.0, "the quick brown fox jumps over the lazy dog"),
        ("toy_2", 1.5, 520.0, "automatic speech recognition with large language models"),
        ("toy_3", 3.0, 330.0, "an embarrassingly simple approach for speech recognition"),
        ("toy_4", 2.5, 600.0, "modal alignment with a trainable linear projector"),
        ("toy_5", 1.8, 480.0, "deep learning and foundation audio encoders"),
    ]

    records = []
    for key, dur, freq, transcript in samples:
        wav_path = os.path.join(audio_dir, f"{key}.wav")
        generate_wav(wav_path, duration_sec=dur, freq=freq)
        records.append({
            "key": key,
            "source": os.path.join("data", "toy_audio", f"{key}.wav"),
            "target": transcript,
            "prompt": "Transcribe speech to text.",
        })

    train_path = os.path.join(data_dir, "sample_train.jsonl")
    val_path = os.path.join(data_dir, "sample_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in records[:4]:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in records[4:]:
            f.write(json.dumps(r) + "\n")

    print(f"[*] Generated toy dataset:")
    print(f"    Train: {train_path} ({len(records[:4])} samples)")
    print(f"    Val:   {val_path} ({len(records[4:])} samples)")
    print(f"    Audio: {audio_dir}")


if __name__ == "__main__":
    main()
