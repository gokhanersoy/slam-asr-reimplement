# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, **ISSAI Turkish Speech Corpus (ISSAI_TSC_218)** veri seti (~218 saatlik Türkçe konuşma verisi) ve modern Türkçe/çokdilli temel modeller (OpenAI Whisper + Qwen2.5 / Trendyol-LLM) kullanılarak geliştirilmiştir.

---

## 📖 Genel Bakış / Overview

SLAM-ASR mimarisi, büyük ve karmaşık çok modlu ASR modellerine (Q-Former, gated cross-attention veya LLM LoRA ince ayarı) ihtiyaç olmadığını kanıtlar:
* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` (veya `facebook/mms-300m`) — **DONDURULMUŞ (FROZEN)**.
* **Büyük Dil Modeli (LLM Backbone):** `Qwen/Qwen2.5-7B-Instruct` veya `Trendyol/Trendyol-LLM-7b-chat-v1.0` — **DONDURULMUŞ (FROZEN)**.
* **Projektör (Projector):** $k=5$ zaman boyutunda ardışık kare birleştirme (temporal downsampling, 50 Hz $\to$ 10 Hz) ve tek bir Lineer katman — **EĞİTİLEN TEK MODÜL** (~15M - 20M parametre).

Modelin parametrelerinin yalnızca **%0.6'sı** eğitilerek, doğrudan Türkçe transkripsiyon yeteneği kazandırılır.

---

## 📊 Veri Seti: ISSAI Turkish Speech Corpus (ISSAI_TSC_218)

Nazarbayev Üniversitesi Institute of Smart Systems and Artificial Intelligence (ISSAI) tarafından yayınlanan en kapsamlı açık kaynaklı Türkçe ses korpusudur:
* **Hugging Face:** [`issai/Turkish_Speech_Corpus`](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
* **Arşiv Dosyası:** `ISSAI_TSC_218.tar.gz` (~21.4 GB)
* **Ses Süresi:** ~218.2 saat
* **Kayıt Sayısı:** 186.171 ses kaydı (16 kHz WAV)
* **Lisans:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Veri Seti Hazırlığı
`scripts/prepare_issai_tsc.py` betiği, veri setini otomatik olarak indirir, arşivden çıkarır, Türkçe özel karakter normalizasyonunu (`İ -> i`, `I -> ı` vb.) uygular ve eğitim manifestolarını üretir:
```bash
python scripts/prepare_issai_tsc.py --download --output_dir data/issai_tsc
```
Veya yerel diskinizdeki `ISSAI_TSC_218.tar.gz` dosyasını belirtmek için:
```bash
python scripts/prepare_issai_tsc.py --tar_path /yol/ISSAI_TSC_218.tar.gz --output_dir data/issai_tsc
```

---

## 🏗️ Proje Yapısı

```
tr/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # Donanıma göre otomatik optimize olan konfigürasyon
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle anlık test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # FrameDownsampler, LinearProjector, MLPProjector
│   │   ├── speech_encoder.py           # SpeechEncoderWrapper (Whisper, MMS, HuBERT)
│   │   └── slam_model.py               # SLAM_ASR ana modeli (Türkçe istem formatı)
│   ├── data/
│   │   ├── audio_processor.py          # Ses okuma, 16 kHz dönüştürme, Whisper log-mel
│   │   ├── dataset.py                  # JSONL veri kümesi yükleyici
│   │   └── collator.py                 # Dinamik padding ve etiket maskeleme (-100)
│   ├── utils/
│   │   ├── metrics.py                  # Türkçe duyarlı WER, CER ve metin normalizasyonu
│   │   ├── scheduler.py                # Warmup + Constant öğrenme oranı zamanlayıcısı
│   │   └── device_utils.py             # Donanım algılayıcı (CUDA, MPS, CPU, bfloat16, 4-bit)
│   └── train/
│       └── trainer.py                  # ASRTrainer eğitim döngüsü ve early stopping
├── scripts/
│   ├── prepare_issai_tsc.py            # ISSAI_TSC_218 indirme ve manifesto üretme
│   └── create_toy_turkish_dataset.py   # Sentetik Türkçe test verisi üretici
├── tests/                              # Kapsamlı birim testleri (Türkçe normalizasyon dahil)
├── train.py                            # Model eğitim betiği
├── evaluate.py                         # Model değerlendirme betiği (WER/CER)
├── inference.py                        # Tekil ses transkripsiyon betiği
├── run_tests.py                        # Test koşucu
└── requirements.txt                    # Gerekli bağımlılıklar
```

---

## 🚀 Hızlı Başlangıç

### 1. Testleri Koşturma
```bash
python run_tests.py
```

### 2. Hızlı Sentetik Test Verisi Oluşturma (2 saniye)
Büyük veriyi indirmeden kodun çalıştığını doğrulamak için:
```bash
python scripts/create_toy_turkish_dataset.py
```

### 3. Eğitimi Başlatma
* **Donanıma göre otomatik optimize edilen konfigürasyon ile:**
  ```bash
  python train.py --config config/hardware_adaptive_turkish.yaml
  ```
* **Colab T4 veya RTX A4000 üzerinde:**
  ```bash
  python train.py --config config/colab_t4_turkish.yaml
  ```

### 4. Değerlendirme (WER / CER Hesabı)
```bash
python evaluate.py \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint outputs/slam_asr_turkish_adaptive/best_checkpoint \
    --manifest data/issai_tsc/test.jsonl \
    --language tr \
    --beam_size 4
```

### 5. Ses Dosyasını Yazıya Dökme (Inference)
```bash
python inference.py \
    --audio data/toy_audio_tr/tr_toy_001.wav \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint outputs/slam_asr_turkish_adaptive/best_checkpoint/projector.pt \
    --prompt "Konuşmayı metne dönüştürün."
```
