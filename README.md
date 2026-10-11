# SLAM-ASR (PyTorch Reimplementation)

PyTorch reimplementation of the research paper:
> **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity**  
> *Ziyang Ma, Guanrou Yang, Yifan Yang, Zhifu Gao, Jiaming Wang, Zhihao Du, Fan Yu, Qian Chen, Siqi Zheng, Shiliang Zhang, Xie Chen*  
> [arXiv:2402.08846](https://arxiv.org/abs/2402.08846)

---

## 📖 Genel Bakış / Overview

Recent multimodal speech-language models often incorporate complex components such as temporal compression modules, Q-Formers, gated cross-attention layers, and Parameter-Efficient Fine-Tuning (PEFT/LoRA) on the LLM backbone. 

**SLAM-ASR** demonstrates that such complex designs are unnecessary:
- **Speech Encoder:** Pre-trained acoustic foundation model (HuBERT, WavLM, Whisper) — **FROZEN**.
- **LLM Backbone:** Off-the-shelf pre-trained Chat LLM (Vicuna-7B-v1.5, TinyLlama, LLaMA-2) — **FROZEN**.
- **Projector:** A single linear layer (or 2-layer MLP) with temporal downsampling ($k=5$ consecutive frame concatenation) — **THE ONLY TRAINABLE MODULE** (~18M - 21M parameters).

Despite training only the projector, SLAM-ASR achieves **State-of-the-Art (SOTA)** automatic speech recognition performance on the LibriSpeech benchmark, outperforming much larger and more complex multimodal audio models.

---

## 🏆 LibriSpeech Benchmark (Paper Table 5)

| Model | Speech Encoder | LLM Backbone | Trainable Params | test-clean WER (%) | test-other WER (%) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **SLAM-ASR (Flagship)** | HuBERT X-Large (1B) | Vicuna-7B | **Projector Only (~21.5M)** | **1.94** | **3.81** |
| SLAM-ASR | HuBERT Large (300M) | Vicuna-7B | Projector Only (~21.5M) | 2.30 | 4.53 |
| SLAM-ASR | WavLM Large (300M) | Vicuna-7B | Projector Only (~18.9M) | 2.28 | 4.78 |
| SLAM-ASR | Whisper Large-v2 | Vicuna-7B | Projector Only (~26.2M) | 2.58 | 6.47 |
| *Yu et al. (2024)* | Whisper Large | Vicuna-13B | seg-QF + LoRA | 2.35 | 5.21 |
| *Qwen-Audio* | Whisper Large (fine-tuned) | Qwen-7B | Encoder + LLM | 2.10 | 4.60 |
| *SALMONN* | Whisper + BEATs | Vicuna-13B | win-QF + LoRA | 2.18 | 5.09 |
| *Zipformer (NN-ASR)* | Zipformer-Large | — (RNN-T) | Full Model + LM | 1.95 | 4.29 |
| *Conformer (NN-ASR)* | Conformer-Large | — (AED/CTC) | Full Model + LM | 1.90 | 4.10 |

---

## 🏗️ Mimari ve Çalışma Mantığı / Architecture

```
Speech Audio (16 kHz)
       │
       ▼
┌──────────────────────────────────────┐
│  Speech Foundation Encoder (FROZEN)  │  (HuBERT / WavLM / Whisper, ~50 Hz features)
└──────────────────┬───────────────────┘
                   │  H_S ∈ R^{B × T × D_enc}
                   ▼
┌──────────────────────────────────────┐
│  Frame Downsampler (k = 5)           │  (Concatenate 5 consecutive frames: 50 Hz -> 10 Hz)
└──────────────────┬───────────────────┘
                   │  Z_S ∈ R^{B × (T/5) × (5 · D_enc)}
                   ▼
┌──────────────────────────────────────┐
│  Linear / MLP Projector [TRAINABLE]  │  (Maps to LLM hidden dimension D_llm)
└──────────────────┬───────────────────┘
                   │  E_S ∈ R^{B × (T/5) × D_llm}
                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Input Sequence: USER: <E_S> <Prompt> ASSISTANT: <Target Transcript><eos>  │
└──────────────────┬────────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Causal LLM Backbone (FROZEN)        │  (Vicuna-7B-v1.5 / TinyLlama-1.1B)
└──────────────────┬───────────────────┘
                   │
                   ▼
     Cross-Entropy Loss (Target Transcript tokens ONLY, prefix/audio masked with -100)
```

### 1. Temporal Downsampling
Speech features operate at ~50 Hz. To prevent excessively long sequence lengths for the LLM, the downsampler concatenates every $k=5$ consecutive frames:
$$z_i^S = [h_{(i-1)k+1}^S \,\|\, h_{(i-1)k+2}^S \,\|\, \dots \,\|\, h_{ik}^S]$$
This reduces the temporal frequency to **10 Hz** (100 ms per speech token).

### 2. Projector
- **Linear Projector (Standard):**
  $$\mathbf{E^S} = \mathbf{W} \mathbf{Z^S} + \mathbf{b}$$
- **MLP Projector:**
  $$\mathbf{E^S} = \mathbf{W_2} \cdot \text{ReLU}(\mathbf{W_1} \mathbf{Z^S} + \mathbf{b_1}) + \mathbf{b_2}$$

### 3. Prompt & Multimodal Input Format
- Training format:
  `USER: <S> <P> ASSISTANT: <T><eos>`
  Where `<S>` represents speech embeddings, `<P>` is the prompt (`"Transcribe speech to text."`), and `<T>` is the ground-truth transcript.
- Loss is computed **strictly on transcript tokens** $\langle T \rangle$ and $\langle\text{eos}\rangle$. All preceding prefix, audio, and prompt tokens are masked with label `-100`.
- Inference format:
  `USER: <S> <P> ASSISTANT:`
  Autoregressive decoding with beam search ($b=4$).

---

## 📂 Proje Yapısı / Repository Structure

```
slam-asr-reimplement/
├── config/
│   ├── train_hubert_vicuna.yaml     # Flagship paper configuration (HuBERT + Vicuna-7B)
│   ├── train_wavlm_vicuna.yaml      # WavLM-Large + Vicuna-7B
│   ├── train_whisper_vicuna.yaml    # Whisper-Large-v2 + Vicuna-7B (30s mel padding)
│   ├── train_tinyllama_hubert.yaml  # Lightweight config (<8 GB VRAM)
│   └── train_toy_debug.yaml         # Synthetic toy data config
├── slam_asr/
│   ├── models/
│   │   ├── projector.py             # FrameDownsampler, LinearProjector, MLPProjector
│   │   ├── speech_encoder.py        # SpeechEncoderWrapper (HuBERT, WavLM, Whisper)
│   │   └── slam_model.py            # Unified SLAM_ASR model architecture
│   ├── data/
│   │   ├── audio_processor.py       # Audio loading, resampling to 16kHz, Whisper mel extraction
│   │   ├── dataset.py               # ASRDataset (JSONL and HuggingFace datasets)
│   │   └── collator.py              # ASRDataCollator (dynamic padding & mask creation)
│   ├── utils/
│   │   ├── metrics.py               # LibriSpeech text normalization, Levenshtein, WER & CER
│   │   └── scheduler.py             # Warmup + Constant / Cosine learning rate scheduler
│   └── train/
│       └── trainer.py               # ASRTrainer with early stopping and AMP support
├── scripts/
│   ├── prepare_librispeech_jsonl.py # Manifest generation for LibriSpeech
│   └── create_toy_dataset.py        # Generates synthetic WAVs and manifests for testing
├── tests/
│   ├── test_metrics.py              # WER, CER, Levenshtein distance unit tests
│   ├── test_projector.py            # Downsampler & Projector shape & behavior tests
│   ├── test_collator.py             # Dynamic batching unit tests
│   └── test_slam_model.py           # Forward pass, gradient isolation & label masking tests
├── train.py                         # Training entrypoint
├── evaluate.py                      # Evaluation entrypoint (WER calculation)
├── inference.py                     # Single-audio transcription script
├── run_tests.py                     # Test runner
├── requirements.txt                 # Dependencies
└── pyproject.toml                   # Package specification
```

---

## 🚀 Kurulum / Installation

```bash
# 1. Virtual environment oluşturun
python3 -m venv .venv
source .venv/bin/activate

# 2. Gereksinimleri yükleyin
pip install -r requirements.txt
```

---

## 🧪 Testleri Çalıştırma / Running Unit Tests

Tüm birim testleri çalıştırmak için:
```bash
python3 run_tests.py
```

---

## ⚡ Hızlı Başlangıç / Quickstart

### 1. Test Verisi Oluşturma (Toy Dataset)
```bash
python3 scripts/create_toy_dataset.py
```
Bu komut `data/toy_audio/` klasörüne sentetik WAV dosyaları ve `data/sample_train.jsonl` / `sample_val.jsonl` manifestolarını oluşturur.

### 2. Eğitimi Başlatma / Training
```bash
# Hafif TinyLlama + HuBERT modeliyle eğitim
python3 train.py --config config/train_tinyllama_hubert.yaml

# Veya paper'daki en iyi HuBERT + Vicuna-7B modeliyle:
python3 train.py --config config/train_hubert_vicuna.yaml
```

### 3. Değerlendirme (WER Hesabı) / Evaluation
```bash
python3 evaluate.py \
    --config config/train_hubert_vicuna.yaml \
    --checkpoint outputs/slam_asr_hubert_vicuna/best_checkpoint \
    --manifest data/test-clean.jsonl \
    --beam_size 4
```

### 4. Tek Bir Ses Dosyasını Transcribe Etme / Inference
```bash
python3 inference.py \
    --audio data/toy_audio/toy_1.wav \
    --config config/train_hubert_vicuna.yaml \
    --checkpoint outputs/slam_asr_hubert_vicuna/best_checkpoint/projector.pt \
    --prompt "Transcribe speech to text." \
    --beam_size 4
```

---

## 📊 LibriSpeech Veri Hazırlığı / LibriSpeech Data Prep

LibriSpeech veri setini SLAM-ASR JSONL formatına dönüştürmek için:

```bash
# Yerel klasörden
python3 scripts/prepare_librispeech_jsonl.py \
    --mode local \
    --librispeech_dir /path/to/LibriSpeech \
    --split train-clean-100 \
    --output_jsonl data/train.jsonl

# Veya Hugging Face datasets üzerinden
python3 scripts/prepare_librispeech_jsonl.py \
    --mode hf \
    --split validation.clean \
    --output_jsonl data/dev-clean.jsonl
```

Format:
```json
{"key": "1001-134707-0000", "source": "/path/to/1001-134707-0000.wav", "target": "transcription text here"}
```

---

## ⚙️ Eğitim Detayları / Training Hyperparameters

Makaledeki (Bölüm 3.3) eğitim parametreleri birebir uygulanmıştır:
- **Optimizer:** `AdamW` (learning rate: $1 \times 10^{-4}$, weight decay: $0.0$).
- **Scheduler:** İlk 1,000 adımda lineer warmup, ardından sabit maksimum öğrenme hızı.
- **Maksimum Adım:** 100,000 adım (val loss erken durdurma ile).
- **Batch Size:** HuBERT / WavLM için 6, Whisper için 4 (30 saniye padding).
- **Eğitilebilir Parametreler:** Yalnızca Projector eğitilir (Encoder ve LLM dondurulmuştur).

---

## 📜 Lisans & Atıf / Citation

```bibtex
@article{ma2024embarrassingly,
  title={An Embarrassingly Simple Approach for LLM with Strong ASR Capacity},
  author={Ma, Ziyang and Yang, Guanrou and Yang, Yifan and Gao, Zhifu and Wang, Jiaming and Du, Zhihao and Yu, Fan and Chen, Qian and Zheng, Siqi and Zhang, Shiliang and Chen, Xie},
  journal={arXiv preprint arXiv:2402.08846},
  year={2024}
}
```
