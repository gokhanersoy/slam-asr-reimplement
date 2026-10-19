# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, **ISSAI Turkish Speech Corpus (ISSAI_TSC_218)** veri seti (~218 saatlik Türkçe konuşma verisi) ve modern Türkçe/çokdilli temel modeller (OpenAI Whisper + Qwen2.5 / Trendyol-LLM) kullanılarak geliştirilmiştir.

---

## 📖 Genel Bakış / Overview (t3-ng-ta-gormez: Görmez Multi-Branch Hybrid Projector)

Bu klasör (`t3-ng-ta-gormez`), **Yasin Görmez'in (PeerJ 2024, *DeepTurkish_Hybrid*)** paralel çok kollu derin mimarisi ile serinin en başarılı modeli olan **Hiyerarşik N-Gram Temporal Attention (`t2-ng-ta`: %12.36 WER)** mekanizmasını sentezler:

* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` — **DONDURULMUŞ (FROZEN)**.
* **Büyük Dil Modeli (LLM Backbone):** `Qwen/Qwen2.5-7B-Instruct` — **DONDURULMUŞ (FROZEN)**.
* **Görmez Hiyerarşik Çok Kollu Projektör (`GormezHybridAttentionProjector`):**
  1. **Stage 1 (Acoustic N-Gram Convolutions):**
     Çok dereceli akustik n-gram derinlikli ayrılabilir konvolüsyonlar ($K \in \{1, 3, 5, 9\}$) ile ham 50 Hz karelerde Türkçe sondan eklemeli morfo-fonetik geçişleri ve koartikülasyonu modeller.
  2. **Stage 2 (Segmental Temporal Attention Downsampling, $k=5$):**
     50 Hz kareleri 10 Hz'e sıkıştırırken pencere içi öğrenilmiş dikkat ağırlık merkezi ($c$) ve çerçeve birleştirme (intra-window centroid pooling) uygular.
  3. **Stage 3 (Görmez Çok Kollu Paralel İşleme - Multi-Branch Processing):**
     Sıkıştırılmış akustik tokenlar (10 Hz) paralel üç koldan geçirilir:
     - **Kol 1 (Recurrent Acoustic Memory):** Çift yönlü GRU (`GormezBiGRUMemoryBlock`, hidden=256, out=512) ile konuşma dalga boyu momentumunu ve sürekli fonetik akışı modeller (Görmez 2024 stili).
     - **Kol 2 (Contextual Temporal Attention):** Dual-scale split-heads attention (dim=512, 4 heads) ile yerel pencere kısıtları ve global ünlü uyumu/prosodi bağlamını yakalar (`t2-ng-ta` stili).
     - **Kol 3 (Residual Identity Shortcut):** Doğrudan projeksiyon + LayerNorm + GELU (dim=512) ile ham akustik özellik kalitesini ve geri yayılım gradyan akışını korur.
  4. **Stage 4 (Multi-Branch Fusion & LLM Mapping):**
     `[h_rec (512), h_attn (512), h_res (512)]` kollarını birleştirir ($1536$ dim) $\to$ LayerNorm $\to$ MLP(1536, 2048) $\to$ GELU $\to$ Linear(2048, 3584) ile Qwen LLM gömme uzayına eşler.

---

## 📊 Veri Seti: ISSAI Turkish Speech Corpus (ISSAI_TSC_218)

* **Hugging Face:** [`issai/Turkish_Speech_Corpus`](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
* **Arşiv Dosyası:** `ISSAI_TSC_218.tar.gz` (~21.4 GB)
* **Ses Süresi:** ~218.2 saat (186.171 kayıt, 16 kHz WAV)
* **Lisans:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 🏗️ Proje Yapısı

```
tr/tsc/t3-ng-ta-gormez/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # Görmez Multi-Branch Hibrit konfigürasyonu
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle anlık test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # GormezBiGRUMemoryBlock & GormezHybridAttentionProjector
│   │   ├── slam_model.py               # SLAM_ASR & SLAMASRConfig
│   │   └── speech_encoder.py           # Whisper / HuBERT wrapper
│   ├── data/
│   │   ├── audio_processor.py          # 16kHz resample & 128-bin log-Mel
│   │   ├── collator.py                 # Dinamik padding & collation
│   │   └── dataset.py                  # PyTorch JSONL Dataset
│   └── utils/
│       ├── device_utils.py             # Donanım tespit ve hiperparametre optimizasyonu
│       ├── metrics.py                  # Türkçe WER ve CER hesaplayıcı
│       └── scheduler.py                # Linear warmup + cosine decay
├── scripts/
│   ├── prepare_issai_tsc.py            # ISSAI veri seti hazırlama betiği
│   ├── prepare_common_voice.py         # Mozilla Common Voice hazırlama betiği
│   └── create_toy_turkish_dataset.py   # Test verisi üretici
├── tests/                              # Birim testler (test_projector, test_slam_model vb.)
├── train.py                            # Ana eğitim betiği
├── evaluate.py                         # Test seti WER/CER değerlendirme betiği
├── inference.py                        # Tekil ses dosyası çıkarım betiği
├── slam-asr-tsc.zip                    # Taşınabilir kod arşivi
└── slam_asr_colab_tr_tsc_t3_ng_ta_gormez.ipynb # Google Colab tek tıkla çalıştırma defteri
```
