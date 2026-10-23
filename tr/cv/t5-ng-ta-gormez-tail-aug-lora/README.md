# SLAM-ASR: Türkçe Mozilla Common Voice Konuşma Tanıma (t5-ng-ta-gormez-tail-aug-lora)
### Görmez Çok Kollu Hibrit Projektör + Kuyruk Veri Artırma + PEFT LoRA — Nihai Tepe Modeli (Capstone Model)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Mozilla Common Voice (Türkçe)** veri seti üzerindeki en kapsamlı uyarlamasıdır.

Bu proje (`cv/t5-ng-ta-gormez-tail-aug-lora`), bağımsız olarak test edilen ve rekor kıran **üç temel başarı faktörünü** Common Voice üzerinde birleştirir:
1. **Mimari Güç:** Görmez Hiyerarşik Çok Kollu Hibrit Projektör (N-Gram Konvolüsyon + Segmental Zamansal Dikkat + Çift Yönlü GRU Akustik Hafıza + Residual Shortcut).
2. **Dil Modeli Adaptasyon Gücü:** Qwen2.5-7B üzerine entegre edilen PEFT LoRA ($r=16, \alpha=32$).
3. **Akustik Dayanıklılık Gücü:** Konuşma sonundaki morfolojik eklerin düşmesini ve nefes azalmasını modelleyen hedefli kuyruk veri artırması (**Tail Speedup 1.15x-1.35x + Tail Energy Decay -3dB/-8dB + Tail SpecMasking**).

---

## 🎯 2x2 İki Yönlü Çapraz Korpus Transfer Matrisi (Bidirectional Benchmark Matrix)

| Eğitim Verisi (Training Corpus) | Model Mimarisi | Test: ISSAI_TSC (Spontane Yayın) | Test: Common Voice (Okuma Metni) |
| :--- | :--- | :---: | :---: |
| **ISSAI_TSC** | `t0` (Standart Doğrusal Baseline) | %12.98 WER (%3.72 CER) | %14.85 WER (%3.92 CER) |
| **ISSAI_TSC** | `t4-ng-ta-gormez-lora` | %11.53 WER (%3.27 CER) | %13.16 WER (%3.34 CER) |
| **ISSAI_TSC** | `t5-ng-ta-gormez-tail-aug-lora` | **%11.43 WER (%3.21 CER)** *(ISSAI SOTA)* | **%12.75 WER (%3.17 CER)** *(Zero-Shot SOTA)* |
| **Common Voice** | `t0` (Standart Doğrusal Baseline) | %29.60 WER (%15.22 CER) | %16.02 WER (%4.78 CER) |
| **Common Voice** | `t4-ng-ta-gormez-lora` | %24.16 WER (%11.38 CER) | %13.76 WER (%3.93 CER) |
| **Common Voice** | `t5-ng-ta-gormez-tail-aug-lora` | **%21.69 WER (%8.78 CER)** *(Büyük Sıçrama)* | **%13.41 WER (%3.81 CER)** *(CV SOTA)* |

---

## 🏗️ Mimari ve Veri Bileşenleri Detayı

### 1. Akustik Kodlayıcı (Speech Encoder)
* `openai/whisper-large-v3` (1280 dim, [DONDURULMUŞ / FROZEN]). Ham konuşma sinyalinden 50 Hz log-mel filtre bankası öznitelikleri çıkarır.

### 2. Görmez Hiyerarşik Çok Kollu Hibrit Projektör (`GormezHybridAttentionProjector`)
* **Stage 1 (Akustik N-Gram Konvolüsyonu):** Çok dereceli ($K \in \{1, 3, 5, 9\}$) pencereler ile Türkçe sondan eklemeli morfo-fonetik geçişleri modeller.
* **Stage 2 (Segmental Zamansal Dikkat Havuzlama, $k=5$):** 50 Hz kareleri 10 Hz'e sıkıştırırken pencere içi öğrenilmiş ağırlık merkezi havuzlaması (centroid pooling) uygular.
* **Stage 3 (Görmez Çok Kollu Paralel İşleme):**
  - **Kol 1 (Recurrent Acoustic Memory):** Çift yönlü GRU (`GormezBiGRUMemoryBlock`, hidden=256, out=512) ile konuşma dalga boyu momentumunu ve sürekli fonetik akışı modeller.
  - **Kol 2 (Contextual Temporal Attention):** Dual-scale split-heads dikkat (dim=512, 4 heads) ile yerel pencere kısıtları ve global ünlü uyumu/prosodi bağlamını yakalar.
  - **Kol 3 (Residual Identity Shortcut):** Doğrudan projeksiyon + LayerNorm + GELU (dim=512) ile ham akustik özellik kalitesini korur.
* **Stage 4 (Füzyon ve MLP Projeksiyonu):** `[h_rec, h_attn, h_res]` kollarını birleştirir ($1536$ dim) $\to$ LayerNorm $\to$ MLP $\to$ Qwen LLM'in $3584$ boyutlu gömme uzayına eşler.

### 3. Büyük Dil Modeli ve PEFT LoRA İnce Ayarı
* `Qwen/Qwen2.5-7B-Instruct` temel modeli (4-bit NF4 kuantizasyon).
* Dikkat katmanlarına (`q_proj`, `v_proj`, `k_proj`, `o_proj`) $r=16, \alpha=32, \text{dropout}=0.05$ LoRA adaptörleri eklenir.

### 4. Hedefe Yönelik Kuyruk Veri Artırma Boru Hattı (`tail_augmentations.py`)
* **`TailSpeedPerturbation`:** Cümlenin son $\%20 - \%35$'lik kuyruk dilimi $1.15\times - 1.35\times$ hızlandırılır (40 ms linear cross-fade ile dikiş izi / klik sesi önlenir).
* **`TailEnergyDecay`:** Nefes tükenmesini ve fısıltılı konuşmayı simüle etmek için kuyruk genliğine $-3\text{ dB}$ ila $-8\text{ dB}$ sönümleme uygulanır.
* **`TailSpecMasking`:** Yalnızca aktif konuşma kuyruğunun Mel spektrogram karelerinde zaman ve frekans maskelemesi yoğunlaştırılır.

---

## 📊 Veri Seti: Mozilla Common Voice (Türkçe)

* **Platform:** [Mozilla Data Collective](https://mozilladatacollective.com)
* **Dataset ID:** `cmu5wkah500c2o10719lem8m8`
* **Format:** MP3 sesler 16 kHz mono WAV formatına dönüştürülür, Türkçe metin normalizasyonu uygulanır.

---

## 🏗️ Proje Yapısı

```
tr/cv/t5-ng-ta-gormez-tail-aug-lora/
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
│   │   ├── audio_processor.py          # 16kHz resample, 128-bin log-Mel & Tail Augmentation
│   │   ├── tail_augmentations.py       # Tail speedup, energy decay, specmasking
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
├── tests/                              # Birim testler (test_tail_augmentations, test_projector vb.)
├── train.py                            # Ana eğitim betiği (Tail Aug + LoRA + Görmez ortak eğitimi)
├── evaluate.py                         # Test seti WER/CER değerlendirme betiği (LoRA otomatik algılama)
├── inference.py                        # Tekil ses dosyası çıkarım betiği (LoRA otomatik algılama)
├── slam-asr-cv.zip                     # Taşınabilir kod arşivi
└── slam_asr_colab_tr_cv_t5_ng_ta_gormez_tail_aug_lora.ipynb # Google Colab tek tıkla çalıştırma defteri
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
