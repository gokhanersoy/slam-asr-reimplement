# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, **ISSAI Turkish Speech Corpus (ISSAI_TSC_218)** veri seti (~218 saatlik Türkçe konuşma verisi) ve modern Türkçe/çokdilli temel modeller (OpenAI Whisper + Qwen2.5 / Trendyol-LLM) kullanılarak geliştirilmiştir.

---

## 📖 Genel Bakış / Overview (t2-ng-ta: Hierarchical NGram-Temporal Hybrid Attention)

Bu klasör (`t2-ng-ta`), serinin en başarılı iki mimarisi olan **N-Gram Attention (`t1-ng`: %12.49 WER)** ve **Temporal Attention (`t1-ta`: %12.56 WER)** mekanizmalarını hiyerarşik bir hibrit yapıda bir araya getirir:

* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` — **DONDURULMUŞ (FROZEN)**.
* **Büyük Dil Modeli (LLM Backbone):** `Qwen/Qwen2.5-7B-Instruct` — **DONDURULMUŞ (FROZEN)**.
* **Hiyerarşik Hibrit Projektör (`NGramTemporalHybridProjector`):**
  1. **Stage 1: Multi-Order Continuous Acoustic N-Gram Convolutions ($K \in \{1, 3, 5, 9\}$):**
     50 Hz ham konuşma çerçevelerinde anlık kare (~20ms), difon geçişi (~60ms), hece çekirdeği (~100ms) ve hece öbeği (~180ms) koartikülasyon dinamiklerini derinlemesine konvolüsyonlarla çıkarır ve dinamik softmax kapılama ile birleştirir.
  2. **Stage 2: Segmental Temporal Attention Downsampling ($k=5$):**
     N-gram zenginleştirilmiş akustik kareleri $k=5$'lik segment pencerelerine ayırır. Segment içi öğrenilmiş intra-window soft attention ile akustik ağırlık merkezi ($c$) çıkarılır ve tüm çerçevelerle birleştirilerek hibrit vektör elde edilir ($(k+1) \times D_{enc} = 6 \times 1024 = 6144$).
  3. **Stage 3: Base Linear Embedding Projection:**
     $6144$ boyutlu hibrit vektörü LLM gömme boyutuna ($3584$ veya $4096$) projekte eder.
  4. **Stage 4: Hierarchical Hybrid Temporal-NGram Attention Block (`split_heads`):**
     10 Hz sıkıştırılmış akustik tokenlar üzerinde çok başlıklı dikkat işletir:
     - **Başların %50'si (Local Heads):** $W=5$ tokenlık kayan n-gram pencere maskesi ile yerel hece ve ek kısıtlarını modeller (~500 ms kelime bağlamı).
     - **Başların %50'si (Global Heads):** Tüm cümle boyunca kısıtsız global dikkat kurarak ünlü uyumu, sözdizimi ve ahenk ilişkilerini yakalar.

---

## 📊 Veri Seti: ISSAI Turkish Speech Corpus (ISSAI_TSC_218)

* **Hugging Face:** [`issai/Turkish_Speech_Corpus`](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
* **Arşiv Dosyası:** `ISSAI_TSC_218.tar.gz` (~21.4 GB)
* **Ses Süresi:** ~218.2 saat (186.171 kayıt, 16 kHz WAV)
* **Lisans:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 🏗️ Proje Yapısı

```
tr/tsc/t2-ng-ta/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # Hiyerarşik Hibrit Attention konfigürasyonu
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle anlık test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # Hierarchical NGramTemporalHybridProjector
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
└── slam_asr_colab_tr_tsc_t2_ng_ta.ipynb # Google Colab tek tıkla çalıştırma defteri
```
