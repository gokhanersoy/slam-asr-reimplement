# SLAM-ASR: Türkçe Konuşma Tanıma - Mozilla Common Voice

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Mozilla Common Voice Türkçe** veri seti ile uçtan uca uyarlamasıdır.

Bu çalışma, **Mozilla Data Collective (MDC)** üzerinden sağlanan Common Voice Türkçe ses korpusu ve modern temel modeller (**OpenAI Whisper + Qwen2.5 / Trendyol-LLM**) kullanılarak geliştirilmiştir.

---

## 📖 Genel Bakış / Overview

SLAM-ASR mimarisi, büyük ve karmaşık çok modlu ASR modellerine (Q-Former, gated cross-attention veya LLM LoRA ince ayarı) ihtiyaç olmadığını kanıtlar:
* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-small` — **DONDURULMUŞ (FROZEN)**.
* **Büyük Dil Modeli (LLM Backbone):** `Qwen/Qwen2.5-7B-Instruct` veya `Trendyol/Trendyol-LLM-7b-chat-v1.0` — **DONDURULMUŞ (FROZEN)**.
* **Projektör (Projector):** $k=5$ zaman boyutunda ardışık kare birleştirme (temporal downsampling, 50 Hz $\to$ 10 Hz) ve tek bir Lineer katman — **EĞİTİLEN TEK MODÜL** (~15M - 20M parametre).

Modelin parametrelerinin yalnızca **%0.6'sı** eğitilerek, doğrudan Türkçe transkripsiyon yeteneği kazandırılır.

---

## 📊 Veri Seti: Mozilla Common Voice Türkçe (MDC)

Mozilla Data Collective API entegrasyonu ile otomatik indirilebilir ve hazırlanabilir:
* **MDC Dataset ID:** `cmu5wkah500c2o10719lem8m8`
* **MDC API Endpoint:** `https://mozilladatacollective.com/api/datasets/cmu5wkah500c2o10719lem8m8/download`
* **Format:** MP3 ses kayıtları (`clips/`) ve TSV etiket dosyaları (`train.tsv`, `dev.tsv`, `test.tsv`)
* **Lisans:** Creative Commons CC-0 / Mozilla Public License

### MDC API Anahtarı ve Veri Seti Hazırlığı
Google Colab ortamında sol paneldeki **Secrets (🔑)** menüsüne `MDC_API_KEY` adıyla Mozilla Data Collective API anahtarınızı eklemeniz yeterlidir.

Manuel veya terminal üzerinden çalıştırmak için:
```bash
# Ortam değişkeni ile
export MDC_API_KEY="mdc_api_key_buraya"
python scripts/prepare_common_voice.py --download --output_dir data/cv

# veya doğrudan parametre ile
python scripts/prepare_common_voice.py --download --api_key "mdc_api_key_buraya" --output_dir data/cv

# veya yerel diskinizdeki .tar.gz arşivi ile
python scripts/prepare_common_voice.py --tar_path /yol/common_voice_tr.tar.gz --output_dir data/cv
```

Bu betik:
1. MDC API üzerinden indirme bağlantısını çözer ve arşivi akış halinde indirir.
2. Arşivi `data/cv/extracted` klasörüne açar.
3. `train.tsv`, `dev.tsv` ve `test.tsv` dosyalarını okuyarak Türkçe karakter duyarlı normalizasyonu (`İ -> i`, `I -> ı` vb.) uygular.
4. `train.jsonl`, `val.jsonl` ve `test.jsonl` manifestolarını üretir.

---

## 🏗️ Proje Yapısı

```
tr/cv/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # Donanıma göre otomatik optimize olan konfigürasyon
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
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
│   │   ├── device_utils.py             # Donanıma duyarlı otomatik profilleyici
│   │   └── scheduler.py                # Linear warmup + cosine decay
│   └── train/
│       └── trainer.py                  # PyTorch eğitim döngüsü (AMP, eval, checkpointing)
├── scripts/
│   ├── prepare_common_voice.py         # Mozilla Data Collective API ve veri hazırlama
│   ├── create_toy_turkish_dataset.py   # Hızlı doğrulama için sentetik ses ve manifesto üretici
│   └── make_cv_notebook.py             # Colab notebook üreteci
├── tests/                              # Kapsamlı birim test paketi (33 test)
├── train.py                            # Ana eğitim betiği
├── evaluate.py                         # Test seti üzerinde WER/CER değerlendirme betiği
├── inference.py                        # Tekil ses dosyası çıkarım betiği
├── run_tests.py                        # Birim test çalıştırıcı
└── slam_asr_colab_tr_cv.ipynb          # Google Colab uçtan uca eğitim & çıkarım defteri
```

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
pip install -r requirements.txt
```

### 2. Birim Testleri Çalıştırma
```bash
python run_tests.py
```

### 3. Pipeline Doğrulama (Sentetik Veri ile 30 Saniye)
```bash
python scripts/create_toy_turkish_dataset.py
python train.py --config config/train_toy_debug_turkish.yaml
```

### 4. Common Voice Eğitimi
```bash
# Donanımı otomatik algılayıp en yüksek verim profilini seçer (H100/A100/L4/T4/CPU)
python train.py --config config/hardware_adaptive_turkish.yaml
```

### 5. Değerlendirme (WER / CER)
```bash
python evaluate.py \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint_path checkpoints/hardware_adaptive_turkish/best_checkpoint.pt \
    --test_jsonl data/cv/test.jsonl
```

### 6. Canlı Mikrofon ve Çıkarım (Inference)
Google Colab notebook'undaki (`slam_asr_colab_tr_cv.ipynb`) 10. hücre aracılığıyla:
1. Model VRAM'e bir kez önceden yüklenir.
2. Web tarayıcınızın mikrofonu açılarak canlı Türkçe ses kaydı alınır.
3. 1-2 saniye içinde ekranınızda doğru Türkçe transkripsiyon görüntülenir.
