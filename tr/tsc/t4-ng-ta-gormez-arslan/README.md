# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, **ISSAI Turkish Speech Corpus (ISSAI_TSC_218)** veri seti (~218 saatlik Türkçe konuşma verisi) ve modern Türkçe/çokdilli temel modeller (OpenAI Whisper + Qwen2.5 / Trendyol-LLM) kullanılarak geliştirilmiştir.

---

## 📖 Genel Bakış / Overview (t4-ng-ta-gormez-arslan)

Bu klasör (`t4-ng-ta-gormez-arslan`), projedeki en başarılı akustik projektör mimarisi olan **Görmez Multi-Branch Hybrid (`t3-ng-ta-gormez`: %12.27 WER)** ile Türkçe ASR literatürünün önemli çalışmalarından **Recep Sinan Arslan'ın Doktora Tezi (Gazi Üniversitesi, 2020 - *"Türkçe Konuşma Tanıma Sistemlerinde Çıktı Düzeltme Yöntemleri"*)** çıktı düzeltme (Output Correction) sistemini bir araya getirir.

### 🏛️ Mimarinin İki Temel Ayağı:

#### 1. Akustik Projektör: Görmez Multi-Branch Hybrid (`GormezHybridAttentionProjector`)
* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` — **DONDURULMUŞ (FROZEN)**.
* **Büyük Dil Modeli (LLM Backbone):** `Qwen/Qwen2.5-7B-Instruct` — **DONDURULMUŞ (FROZEN)**.
* **Stage 1 (Akustik N-Gram Konvolüsyonu):** Ham 50 Hz karelerde 1, 3, 5, 9 ardışık akustik koartikülasyon pencereleri ve dinamik softmax yönlendirme.
* **Stage 2 (Segmental Zamansal Dikkat Havuzlama, k=5):** 50 Hz kareleri 10 Hz'e sıkıştırırken intra-window soft attention ağırlık merkezi ($c$) ve çerçeve birleştirme.
* **Stage 3 (Görmez Çok Kollu Paralel İşleme):** Sıkıştırılmış 10 Hz akustik tokenlar paralel 3 koldan geçirilir:
  - **Kol 1 (Recurrent Acoustic Memory):** Çift yönlü GRU (`GormezBiGRUMemoryBlock`, hidden=256, out=512) ile konuşma dalga boyu sürekliliğini ve fonetik akışı modeller (Görmez 2024 stili).
  - **Kol 2 (Contextual Temporal Attention):** Dual-scale split-heads attention (dim=512, 4 heads) ile yerel pencere kısıtları ve global ünlü uyumu/prosodi bağlamını yakalar (`t2-ng-ta` stili).
  - **Kol 3 (Residual Identity Shortcut):** Doğrudan projeksiyon + LayerNorm + GELU (dim=512) ile ham akustik özellik kalitesini ve gradyan akışını korur.
* **Stage 4 (Multi-Branch Fusion & LLM Mapping):** `[h_rec (512), h_attn (512), h_res (512)]` kollarını birleştirir ($1536$ dim) $\to$ LayerNorm $\to$ MLP(1536, 2048) $\to$ GELU $\to$ Linear(2048, 3584) ile Qwen LLM gömme uzayına eşler.

#### 2. Çıktı Düzeltme Modülü: Recep Sinan Arslan Output Correction (`ArslanOutputCorrector`)
* **Sözlük Doğrulama (Lexicon Validation):** Korpusta (179.258 cümlelik eğitim seti + temel sözlük) bulunan ve kurallı Türkçe çekim eklerini taşıyan kelimeler ASLA bozulmaz.
* **Normalized Levenshtein (NL) Mesafesi:**
  $$\text{NL}(w_1, w_2) = \frac{\text{Levenshtein}(w_1, w_2)}{\max(\text{len}(w_1), \text{len}(w_2))}$$
* **Arslan Altın Kuralları (Doktora Tezi, 2020):**
  - **Uzaklık Eşiği ($\theta \le 0.33$):** Yalnızca $\le 0.33$ mesafedeki adaylar dikkate alınır (kelimenin bambaşka bir kelimeye dönüşmesi engellenir).
  - **Kelime Uzunluğu Kısıtı ($\text{len} > 3$):** 3 veya daha az karakterli kısa kelimelere ("ve", "de", "da", "bu", "o", "mi") asla dokunulmaz.
  - **Frekans Tabanlı Sıralama:** Birden fazla eşit mesafeli aday arasında korpusta en yüksek frekansa sahip kelime seçilir.
* **Konuşma Dili & Yazım Standardizasyonu:**
  - Konuşma dili kısaltmaları: `biliyosun` $\to$ `biliyorsun`, `geliyo` $\to$ `geliyor`, `yapcaz` $\to$ `yapacağız`.
  - Yabancı imla eşlemesi: `coronanın` $\to$ `koronanın`.
  - Ayrı/bitişik ek ayrımı: `zaman da` $\leftrightarrow$ `zamanda`.

---

## 📊 Veri Seti: ISSAI Turkish Speech Corpus (ISSAI_TSC_218)

* **Hugging Face:** [`issai/Turkish_Speech_Corpus`](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
* **Arşiv Dosyası:** `ISSAI_TSC_218.tar.gz` (~21.4 GB)
* **Ses Süresi:** ~218.2 saat (186.171 kayıt, 16 kHz WAV)
* **Lisans:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 🏗️ Proje Yapısı

```
tr/tsc/t4-ng-ta-gormez-arslan/
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
│       ├── arslan_corrector.py         # Recep Sinan Arslan Output Corrector & Normalized Levenshtein
│       ├── device_utils.py             # Donanım tespit ve hiperparametre optimizasyonu
│       ├── metrics.py                  # Türkçe WER ve CER hesaplayıcı
│       └── scheduler.py                # Linear warmup + cosine decay
├── scripts/
│   ├── apply_arslan_correction.py      # Mevcut eval_results.jsonl üzerinde anında düzeltme çalıştırıcı
│   ├── prepare_issai_tsc.py            # ISSAI veri seti hazırlama betiği
│   ├── prepare_common_voice.py         # Mozilla Common Voice hazırlama betiği
│   └── create_toy_turkish_dataset.py   # Test verisi üretici
├── tests/                              # Birim testler (test_arslan_corrector, test_projector, vb.)
├── train.py                            # Ana eğitim betiği
├── evaluate.py                         # Test seti değerlendirme betiği (--use_arslan_correction destekli)
├── inference.py                        # Tekil ses dosyası çıkarım betiği
├── slam-asr-tsc.zip                    # Taşınabilir kod arşivi
└── slam_asr_colab_tr_tsc_t4_ng_ta_gormez_arslan.ipynb # Google Colab tek tıkla çalıştırma defteri
```

---

## ⚡ Hızlı Kullanım / Standalone Çıktı Düzeltme

Önceki modelinizin (`t3-ng-ta-gormez`) ürettiği `eval_results.jsonl` dosyasını GPU çalıştırmadan saniyeler içinde düzeltmek için:

```bash
python scripts/apply_arslan_correction.py \
    --input eval_results.jsonl \
    --output eval_results_arslan_corrected.jsonl \
    --train_manifest data/issai_tsc/train.jsonl \
    --threshold 0.33
```
