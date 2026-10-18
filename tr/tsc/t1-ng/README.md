# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, **ISSAI Turkish Speech Corpus (ISSAI_TSC_218)** veri seti (~218 saatlik Türkçe konuşma verisi) ve modern Türkçe/çokdilli temel modeller (OpenAI Whisper + Qwen2.5 / Trendyol-LLM) kullanılarak geliştirilmiştir.

---

## 📖 Genel Bakış / Overview (t1-ng: N-Gram Attention)

Bu klasör (`t1-ng`), konuşmadaki eşseslenme (coarticulation) dinamiklerini ve yerel dilbilgisel bağlam kısıtlarını modellemek üzere **N-Gram Attention** mekanizmasının entegre edildiği deneysel daldır:
* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` — **DONDURULMUŞ (FROZEN)**.
* **Büyük Dil Modeli (LLM Backbone):** `Qwen/Qwen2.5-7B-Instruct` — **DONDURULMUŞ (FROZEN)**.
* **Projektör (`NGramAttentionProjector`):**
  1. **Multi-Order Acoustic N-Gram Convolutions ($K \in \{1, 3, 5, 9\}$):** 
     Sürekli konuşma sinyalinde farklı akustik n-gram mertebelerini paralel derinlemesine 1D evrişimlerle çıkarır:
     - **1-gram (Unigram, K=1):** Anlık kare bilgisi (~20 ms)
     - **2-gram (Bigram, K=3):** Doğrudan difon geçişleri ve eşseslenme (~60 ms)
     - **3-gram (Trigram, K=5):** Trifon ve hece çekirdeği (~100 ms)
     - **5-gram (5-gram, K=9):** Alt-kelime ve hece kümesi bağlamı (~180 ms)
     Her bir n-gram mertebesi, öğrenilebilir dinamik softmax kapılama (`gate_router`) ile zamansal olarak ağırlıklandırılarak birleştirilir.
  2. **Zamansal Sıkıştırma (Temporal Downsampling):** $k=5$ ardışık n-gram çerçevesinin birleştirilmesi (50 Hz $\to$ 10 Hz).
  3. **Lineer Projeksiyon:** Birleştirilmiş özellikleri LLM gömme boyutuna ($3584$) eşleme.
  4. **Local Windowed N-Gram Contextual Attention ($W=5$ Token):**
     Sıkıştırılmış konuşma tokenları arasında (~100 ms/token) yerel 5-gram kayan pencere maskesi ile self-attention kurarak, uzak karelerin getirebileceği gürültüden arındırılmış yerel n-gram dil modeli kısıtlarını modeller.

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
tr/tsc/t1-ng/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # N-Gram Attention konfigürasyonu
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle anlık test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # MultiOrderNGramConv, NGramContextAttention, NGramAttentionProjector
│   │   ├── speech_encoder.py           # SpeechEncoderWrapper (Whisper, MMS, HuBERT)
│   │   └── slam_model.py               # SLAM_ASR ana modeli (N-Gram parametreleri)
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
├── tests/                              # Kapsamlı birim testleri (N-Gram Attention dahil)
├── train.py                            # Model eğitim betiği
├── evaluate.py                         # Model değerlendirme betiği (WER/CER)
├── inference.py                        # Tekil ses transkripsiyon betiği
├── run_tests.py                        # Test koşucu
├── slam_asr_colab_tr_tsc_t1_ng.ipynb   # Colab eğitim notebook'u
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

---

## 🔬 N-Gram Attention Hipotezi ve Tasarım Mantığı

Doğal konuşmada fonemler tek tek telaffuz edilmez; bir ses kendisinden önceki ve sonraki seslerle birlikte şekillenir (eşseslenme / coarticulation). Standart SLAM-ASR'nin $k=5$ blok birleştirmesi, blok sınırlarında kalan fonemleri sert bir şekilde böler.

N-Gram Attention Projector iki temel avantaj sunar:
1. **Çok Mertebeli Akustik N-Gramlar:** Unigram (20 ms), Bigram (60 ms), Trigram (100 ms) ve 5-gram (180 ms) evrişim dalları sayesinde sesler arasındaki geçişler blok birleştirmeden önce yumuşatılır ve eşseslenme bağlamı zenginleştirilir.
2. **Yerel Kayan Pencere Dikkati:** Sıkıştırma sonrası elde edilen 10 Hz tokenlar üzerinde global dikkat yerine yerel 5-gram token penceresi ($W=5$, ~500 ms) uygulanarak, konuşma sinyalinin yerel kelime/ek diziliş kurallarına odaklanması sağlanır.
