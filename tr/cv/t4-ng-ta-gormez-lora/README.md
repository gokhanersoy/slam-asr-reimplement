# SLAM-ASR: Türkçe Mozilla Common Voice Konuşma Tanıma (t4-ng-ta-gormez-lora)
### Görmez Çok Kollu Hibrit Projektör + PEFT LoRA Adaptasyonu — İki Yönlü Çapraz Korpus Transfer Deneyi

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Mozilla Common Voice (Türkçe)** veri seti üzerindeki uyarlamasıdır.

Bu çalışma, SOTA **Görmez Hiyerarşik Çok Kollu Projektörü** ile dondurulmuş büyük dil modeli (Qwen2.5-7B) üzerine eklenen **PEFT LoRA (Low-Rank Adaptation)** ince ayarını Common Voice (okuma konuşması) üzerinde birleştirerek akademik yayındaki **İki Yönlü Çapraz Korpus Transfer Simetrisini (Bidirectional Cross-Corpus Transfer Symmetry)** test eder.

---

## 🎯 Akademik Hipotez ve Rasyonel (Bidirectional Generalization)

Konuşma tanıma literatüründe modellerin farklı akustik alanlara transfer kabiliyeti kritik bir başarı metriğidir:
* **ISSAI_TSC:** TV/radyo programlarından kesitler, spontane konuşmalar, duraksamalar ("ııııı"), arka plan çakışmaları, dinamik ses şiddeti değişimleri.
* **Mozilla Common Voice:** Mikrofona tane tane okunan, gönüllüler tarafından zenginleştirilmiş temiz okuma metinleri.

### 2x2 Çapraz Korpus Transfer Matrisi:
| Eğitim Verisi (Training Corpus) | Test: ISSAI_TSC (Spontane Yayın) | Test: Common Voice (Okuma Metni) |
| :--- | :---: | :---: |
| **ISSAI (t0 Baseline)** | %12.98 WER (%3.72 CER) | %14.85 WER (%3.92 CER) |
| **ISSAI (t4-lora)** | %11.53 WER (%3.27 CER) | %13.16 WER (%3.34 CER) |
| **ISSAI (t5-tail-aug-lora)** | **%11.43 WER (%3.21 CER)** *(ISSAI SOTA)* | **%12.75 WER (%3.17 CER)** *(Zero-Shot SOTA)* |
| **Common Voice (t0 Baseline)** | %29.60 WER (%15.22 CER) | %16.02 WER (%4.78 CER) |
| **Common Voice (t4-ng-ta-gormez-lora)** | **%24.16 WER (%11.38 CER)** | **%13.76 WER (%3.93 CER)** |
| **Common Voice (t5-ng-ta-gormez-tail-aug-lora)** | **%21.69 WER (%8.78 CER)** *(Büyük Sıçrama)* | **%13.41 WER (%3.81 CER)** *(CV SOTA)* |

Bu deney ile literatüre iki önemli katkı sağlanacaktır:
1. **Mimari Evrensellik:** Görmez + LoRA mimarisinin yalnızca zorlu ve spontane ISSAI verisinde değil, mikrofona okunan Common Voice verisinde de `cv/t0` taban çizgisine kıyasla tutarlı bir kazanç sağlayıp sağlamadığı.
2. **Akustik Alan Asimetrisi Analizi:** Spontane konuşmayla eğitilen modellerin okuma konuşmasına kolay adapte olduğu (%14.85 $\to$ %12.75), ancak okuma konuşmasıyla eğitilen modellerin spontane konuşmaya transferde zorlandığı (%29.60) asimetrisini Görmez + LoRA'nın ne ölçüde kapatabildiği.

---

## 📖 Mimari Bileşenleri

* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` — **DONDURULMUŞ (FROZEN)**. Ham konuşma sinyalinden 50 Hz log-mel özellik çıkarır.
* **Görmez Hiyerarşik Çok Kollu Projektör (`GormezHybridAttentionProjector`):** — **EĞİTİLİR (TRAINABLE)**
  1. **Stage 1 (Akustik N-Gram Konvolüsyonları):** $K \in \{1, 3, 5, 9\}$ ardışık n-gram derinlikli konvolüsyonlarla Türkçe morfo-fonetik geçişleri ve hece sınırlarını modeller.
  2. **Stage 2 (Segmental Zamansal Dikkat Havuzlama, $k=5$):** 50 Hz kareleri 10 Hz'e sıkıştırırken pencere içi öğrenilmiş ağırlık merkezi havuzlaması uygular.
  3. **Stage 3 (Görmez Çok Kollu Paralel İşleme):**
     - **Kol 1 (Recurrent Acoustic Memory):** Çift yönlü GRU (`GormezBiGRUMemoryBlock`, hidden=256, out=512) ile konuşma dalga boyu momentumunu ve sürekli fonetik akışı modeller.
     - **Kol 2 (Contextual Temporal Attention):** Dual-scale split-heads attention (dim=512, 4 heads) ile yerel pencere kısıtları ve global ses uyumu bağlamını yakalar.
     - **Kol 3 (Residual Identity Shortcut):** Doğrudan projeksiyon + LayerNorm + GELU (dim=512) ile ham akustik özellik gradyan akışını korur.
  4. **Stage 4 (Füzyon ve MLP Projeksiyonu):** Üç kol birleştirilir (1536 dim) $\to$ LayerNorm $\to$ MLP(1536, 2048) $\to$ GELU $\to$ Linear(2048, 3584) ile Qwen LLM gömme uzayına eşlenir.
* **Büyük Dil Modeli (LLM Backbone) + PEFT LoRA:** `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) — **LoRA UYARLAMASI EĞİTİLİR (TRAINABLE)**
  - Base model dondurulur.
  - LLM self-attention projeksiyonlarına (`q_proj`, `v_proj`, `k_proj`, `o_proj`) $r=16, \alpha=32, \text{dropout}=0.05$ LoRA adaptörleri entegre edilir.

---

## 📊 Veri Seti: Mozilla Common Voice (Türkçe)

* **Platform:** [Mozilla Data Collective](https://mozilladatacollective.com)
* **Dataset ID:** `cmu5wkah500c2o10719lem8m8`
* **Format:** MP3 sesler 16 kHz mono WAV formatına dönüştürülür, Türkçe metin normalizasyonu uygulanır.

---

## 🏗️ Proje Yapısı

```
tr/cv/t4-ng-ta-gormez-lora/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # Otomatik GPU profilli konfigürasyon (T4, L4, A100, A4000)
│   ├── colab_t4_turkish.yaml           # Colab T4 16GB VRAM için 4-bit NF4 + LoRA konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle 30 sn hızlı test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # GormezBiGRUMemoryBlock & GormezHybridAttentionProjector
│   │   ├── slam_model.py               # SLAM_ASR & SLAMASRConfig (PEFT LoRA desteğiyle)
│   │   └── speech_encoder.py           # Whisper / HuBERT wrapper
│   ├── data/
│   │   ├── audio_processor.py          # 16kHz resample & 128-bin log-Mel
│   │   ├── collator.py                 # Dinamik padding & collation
│   │   └── dataset.py                  # PyTorch JSONL Dataset
│   └── utils/
│       ├── device_utils.py             # Donanım tespit ve hiperparametre optimizasyonu
│       ├── metrics.py                  # Türkçe WER ve CER hesaplayıcı
│       └── scheduler.py                # Linear warmup + constant
├── scripts/
│   ├── prepare_common_voice.py         # Mozilla Data Collective Common Voice indirme ve hazırlama betiği
│   ├── prepare_issai_tsc.py            # Çapraz korpus testi için ISSAI hazırlama betiği
│   └── create_toy_turkish_dataset.py   # Test verisi üretici
├── tests/                              # Birim testler (test_projector, test_slam_model, test_metrics vb.)
├── train.py                            # Ana eğitim betiği (Görmez Projektör + LoRA ortak eğitimi)
├── evaluate.py                         # Test seti WER/CER değerlendirme betiği (LoRA otomatik algılama)
├── inference.py                        # Tekil ses dosyası çıkarım betiği (LoRA otomatik algılama)
├── slam-asr-cv.zip                     # Taşınabilir kod arşivi
└── slam_asr_colab_tr_cv_t4_ng_ta_gormez_lora.ipynb # Google Colab tek tıkla çalıştırma defteri
```

---

## 🚀 Hızlı Başlangıç

### 1. Birim Testleri Çalıştırın
```bash
python run_tests.py
```

### 2. Sentetik Veri ile Hızlı Test (30 Saniye)
```bash
python scripts/create_toy_turkish_dataset.py
python train.py --config config/hardware_adaptive_turkish.yaml --train_manifest data/sample_train_tr.jsonl --val_manifest data/sample_val_tr.jsonl --max_steps 2
```

### 3. Mozilla Common Voice Verisini İndirin
```bash
python scripts/prepare_common_voice.py --download --api_key "MDC_API_KEY" --output_dir data/cv
```

### 4. Eğitimi Başlatın
```bash
python train.py --config config/hardware_adaptive_turkish.yaml
```

### 5. Değerlendirme (In-Domain: Common Voice & Out-of-Domain: ISSAI_TSC)
```bash
# In-Domain (Common Voice)
python evaluate.py \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint outputs/slam_asr_cv_turkish_adaptive/best_checkpoint \
    --manifest data/cv/test.jsonl \
    --language tr

# Out-of-Domain (ISSAI_TSC Spontane Yayın)
python evaluate.py \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint outputs/slam_asr_cv_turkish_adaptive/best_checkpoint \
    --manifest data/issai_tsc/test.jsonl \
    --language tr
```
