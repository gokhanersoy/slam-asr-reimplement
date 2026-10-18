# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, **ISSAI Turkish Speech Corpus (ISSAI_TSC_218)** veri seti (~218 saatlik Türkçe konuşma verisi) ve modern Türkçe/çokdilli temel modeller (OpenAI Whisper + Qwen2.5 / Trendyol-LLM) kullanılarak geliştirilmiştir.

---

## 📖 Genel Bakış / Overview (t1-enhanced: Enhanced Baseline)

Bu klasör (`t1-enhanced`), orijinal **Baseline (`t0`)** modelini modern derin öğrenme yöntemleriyle güçlendiren geliştirilmiş temel çizgidir:

1. **Projektör Tarafı (2-Katmanlı Non-Linear MLP):**
   * Tekil lineer matris yerine `Linear(in_dim, 2048) -> LayerNorm -> GELU/SwiGLU -> Dropout -> Linear(2048, out_dim) -> LayerNorm` yapısı.
   * Esnek zamansal sıkıştırma ($k \in \{4, 5, 6\}$) ve opsiyonel 1D derinlikli ayrışımlı konvolüsyonel downsampling (`conv1d`).
2. **Adaptasyon Tarafı (LLM LoRA / QLoRA):**
   * Dil modelinin (Qwen2.5-7B) dikkat katmanlarına (`q_proj`, `v_proj`, `k_proj`, `o_proj`) $r=16, \alpha=32$ parametre-etkin LoRA adaptörleri eklenir.
   * 7B parametrelik LLM dondurulmuş (frozen) kalırken sadece ~20M LoRA parametresi eğitilir ve model Türkçe ses temsillerine doğrudan adapte olur.
3. **Loss / Çoklu Görev (Yardımcı CTC Kaybı):**
   * Projektör çıkışına eklenen hafif bir Türkçe karakter CTC başlığı ile modelin ses dizilimini zaman boyunca monotonik olarak koruması sağlanır:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LLM}} + 0.2 \cdot \mathcal{L}_{\text{CTC}}$$
   * Bu kayıp, fonetik halüsinasyonları ve kelime yutmalarını engeller.
4. **Kodlayıcı İnce Ayarı (Opsiyonel):**
   * İstenirse Whisper-Large-v3'ün en üstteki $N$ katmanı (`unfreeze_encoder_layers`) eğitilebilir kılınabilir.

---

## 📊 Veri Seti: ISSAI Turkish Speech Corpus (ISSAI_TSC_218)

* **Hugging Face:** [`issai/Turkish_Speech_Corpus`](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
* **Arşiv Dosyası:** `ISSAI_TSC_218.tar.gz` (~21.4 GB)
* **Ses Süresi:** ~218.2 saat (186.171 kayıt, 16 kHz WAV)
* **Lisans:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 🏗️ Proje Yapısı

```
tr/tsc/t1-enhanced/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # MLP + LoRA + CTC konfigürasyonu
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle anlık test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # EnhancedMLPProjector & Conv1dDownsampler
│   │   ├── slam_model.py               # SLAM_ASR (LoRA & Auxiliary CTC entegreli)
│   │   └── speech_encoder.py           # Whisper / HuBERT wrapper
│   ├── data/
│   │   ├── audio_processor.py          # 16kHz resample & 128-bin log-Mel
│   │   ├── collator.py                 # Dinamik padding & collation
│   │   └── dataset.py                  # PyTorch JSONL Dataset
│   ├── train/
│   │   └── trainer.py                  # LoRA & CTC ağırlıklarını kaydeden eğitim yöneticisi
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
├── evaluate.py                         # Test seti WER/CER değerlendirme betiği (LoRA destekli)
├── inference.py                        # Tekil ses dosyası çıkarım betiği (LoRA destekli)
├── slam-asr-tsc.zip                    # Taşınabilir kod arşivi
└── slam_asr_colab_tr_tsc_t1_enhanced.ipynb # Google Colab tek tıkla çalıştırma defteri
```
