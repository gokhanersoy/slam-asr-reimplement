# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)
## t1-feature-aug: Özellik Çıkarımı ve Veri Artırma (Feature Extraction & Augmentation)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, tez danışmanınızın önerdiği **3. Maddeyi (Özellik Çıkarımı ve Veri Artırma / Feature Extraction & Augmentation)** temel model (`t0` baseline) üzerinde hayata geçirir.

---

## 📖 Genel Bakış / Overview (t1-feature-aug)

### 🎯 Amaç ve Motivasyon
Temel SLAM-ASR modelinde (`t0`), sabit ve sakin stüdyo kayıtları (ISSAI TSC) üzerinde eğitim yapıldığında model gerçek hayattaki **konuşma hızı dalgalanmalarına**, **fısıltı veya bağırma gibi dinamik genlik değişimlerine** ve **akustik örtüşme / hece yutulması (*hypoarticulation / lenition*)** durumlarına karşı kırılgan kalabilmektedir.

`t1-feature-aug` çalışması, bu zorlukları aşmak için 3 temel çevrimiçi (online) veri artırma ve özellik zenginleştirme tekniğini sisteme entegre eder:

1. **Hız Değiştirme ile Veri Çoğullama (Speed Perturbation - 0.9x, 1.0x, 1.1x):**
   - Dalga boyunu doğrusal yeniden örnekleme ile 0.9x (yavaşlatılmış), 1.0x (normal) ve 1.1x (hızlandırılmış) oranlarında dinamik olarak esnetir.
   - Cümle sonlarında hızlanan konuşma ve yuvarlanan hecelere karşı modelin zaman ölçeği duyarsızlığı kazanmasını sağlar.
2. **Ses Seviyesi / Kazanç Artırımı (Volume / Gain Perturbation):**
   - Rastgele $\pm 4\text{ dB}$ kazanç değişimi uygulayarak fısıltılı veya bağıran ses kayıtları arasındaki enerji farklarına karşı dayanıklılık kazandırır.
3. **Spektrogram Maskeleme (SpecAugment - Park et al., 2019):**
   - 128-bin Log-Mel spektrogramı üzerinde zaman blokları ($T=35$) ve frekans bantları ($F=15$) rastgele maskelenir.
   - Modelin yutulmuş veya zayıf sesleri tekil karelerden ezberlemek yerine konuşmanın genel dilsel bağlamından çıkarmasını teşvik eder.
4. **Çevrimiçi (Online) Eğitim & Temiz Değerlendirme:**
   - Veri artırma yalnızca eğitim (`is_training=True`) aşamasında devrededir. Validasyon, test ve çıkarım aşamalarında ham ses üzerinden temiz değerlendirme yapılır.

---

## 📊 Veri Seti: ISSAI Turkish Speech Corpus (ISSAI_TSC_218)

* **Hugging Face:** [`issai/Turkish_Speech_Corpus`](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
* **Arşiv Dosyası:** `ISSAI_TSC_218.tar.gz` (~21.4 GB)
* **Ses Süresi:** ~218.2 saat (186.171 kayıt, 16 kHz WAV)
* **Lisans:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 🏗️ Proje Yapısı

```
tr/tsc/t1-feature-aug/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # Donanıma göre optimize veri artırma konfigürasyonu
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle anlık test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # FrameDownsampler, LinearProjector, MLPProjector
│   │   ├── speech_encoder.py           # SpeechEncoderWrapper (Whisper, MMS, HuBERT)
│   │   └── slam_model.py               # SLAM_ASR ana modeli (Whisper + Linear + Qwen)
│   ├── data/
│   │   ├── augmentations.py            # SpeedPerturbation, VolumePerturbation, SpecAugment
│   │   ├── audio_processor.py          # Çevrimiçi veri artırma entegreli ses işleyici
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
│   ├── prepare_common_voice.py         # Mozilla Common Voice hazırlama betiği
│   └── create_toy_turkish_dataset.py   # Sentetik Türkçe test verisi üretici
├── tests/                              # Birim testler (test_augmentations dahil 37 test)
├── train.py                            # Model eğitim betiği (online augmentations devrede)
├── evaluate.py                         # Model değerlendirme betiği (temiz test verisiyle)
├── inference.py                        # Tekil ses transkripsiyon betiği
├── run_tests.py                        # Test koşucu
├── slam-asr-tsc.zip                    # Taşınabilir kod arşivi
└── slam_asr_colab_tr_tsc_t1_feature_aug.ipynb # Google Colab tek tıkla çalıştırma defteri
```
