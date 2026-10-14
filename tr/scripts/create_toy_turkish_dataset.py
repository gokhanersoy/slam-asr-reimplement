#!/usr/bin/env python3
"""Generate synthetic Turkish audio dataset for rapid pipeline verification."""

import json
import math
import os
import struct
import wave

TURKISH_SAMPLE_SENTENCES = [
    ("tr_toy_001", "merhaba dünya bu bir yapay zeka modelidir"),
    ("tr_toy_002", "türkiye cumhuriyeti yüzüncü yılını gururla kutluyor"),
    ("tr_toy_003", "türkçe ses tanıma ve konuşma modeli başarıyla çalışıyor"),
    ("tr_toy_004", "bugün hava çok güzel ve güneş parlıyor"),
    ("tr_toy_005", "derin öğrenme modelleri ses sinyallerini metne dönüştürür"),
    ("tr_toy_006", "doğal dil işleme ve akustik modeller birlikte eğitilir"),
    ("tr_toy_007", "istanbul ve ankara türkiyenin en kalabalık şehirleridir"),
    ("tr_toy_008", "büyük dil modelleri çok dilli yeteneklere sahiptir"),
    ("tr_toy_009", "yapay zeka teknolojisi her geçen gün daha da gelişiyor"),
    ("tr_toy_010", "bu ses kaydı türkçe test manifestosu için üretilmiştir"),
]


def generate_synthetic_wav(
    filepath: str,
    duration_sec: float = 2.5,
    sample_rate: int = 16000,
    freq: float = 440.0,
):
    """Generate a clean 16kHz sine wave audio file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    with wave.open(filepath, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(num_samples):
            # Slightly modulate frequency to mimic harmonic variations
            f = freq * (1.0 + 0.1 * math.sin(2 * math.pi * 2.0 * (i / sample_rate)))
            val = int(16000.0 * math.sin(2.0 * math.pi * f * (i / sample_rate)))
            data = struct.pack("<h", max(-32767, min(32767, val)))
            wav_file.writeframes(data)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = os.path.join(base_dir, "data", "toy_audio_tr")
    os.makedirs(audio_dir, exist_ok=True)

    train_samples = []
    val_samples = []

    print("[*] Generating synthetic Turkish toy dataset...")
    for idx, (key, text) in enumerate(TURKISH_SAMPLE_SENTENCES):
        duration = 2.0 + (idx % 3) * 0.5
        wav_path = os.path.join(audio_dir, f"{key}.wav")
        base_freq = 300.0 + (idx * 35.0)
        generate_synthetic_wav(wav_path, duration_sec=duration, freq=base_freq)

        entry = {
            "key": key,
            "audio_filepath": os.path.abspath(wav_path),
            "text": text,
            "duration": duration,
        }

        # First 8 train, last 2 val
        if idx < 8:
            train_samples.append(entry)
        else:
            val_samples.append(entry)

    train_path = os.path.join(base_dir, "data", "sample_train_tr.jsonl")
    val_path = os.path.join(base_dir, "data", "sample_val_tr.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[*] Created {len(train_samples)} train samples in: {train_path}")
    print(f"[*] Created {len(val_samples)} val samples in:   {val_path}")
    print("[*] Turkish toy dataset generated successfully.")


if __name__ == "__main__":
    main()
