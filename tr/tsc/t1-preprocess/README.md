# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)
## t1-preprocess: Gelişmiş Akustik Ön İşleme (Advanced Acoustic Preprocessing)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, tez danışmanınızın önerdiği **1. Maddeyi (Akustik Ön İşleme / Preprocessing)** temel model (`t0` baseline) üzerinde hayata geçirir.

---

## 📖 Genel Bakış / Overview (t1-preprocess)

### 🎯 Çözülen Akustik ve Fonetik Sorunlar
Konuşma ve şarkı söyleme esnasında cümle/ibare sonlarına doğru nefesin azalmasıyla akciğer altı hava basıncı düşer (*subglottal pressure drop / trailing energy loss*). Bu durum cümle sonlarındaki hecelerin enerjisinin sönümlenmesine ve hızlı konuşmada hecelerin yutulmasına (*hypoarticulation / lenition*) yol açar. Türkçe gibi sondan eklemeli dillerde cümlenin kilit anlamını taşıyan ekler (*-elim*, *-dim*, *-lar*) tam da bu zayıflayan son bölümlerde yer aldığından, standart ASR modellerinde son kelimelerde ve eklerde hatalar kümelenir.

`t1-preprocess` mimarisi bu sorunu 4 temel ön işleme ve çerçeveleme adımıyla çözer:

1. **Dinamik Aralık Sıkıştırma (Dynamic Range Compression - DRC):**
   - 25 ms hareketli pencere ile sinyalin RMS enerji örtüsü (*envelope follower*) çıkarılır.
   - Tepe sesler yumuşak bir oranla sıkıştırılırken, enerjisi düşen fısıltılı ve zayıf cümle sonu hecelerinin genliği 2-3 kat artırılır; kırpılma (*clipping*) kesinlikle önlenir.
2. **Akustik Kuyruk Tamponu (Trailing Silence Buffer, 300 ms):**
   - Ses sinyalinin sonuna yapay 300 ms sessizlik eklenir. Whisper'ın konvolüsyonel pencereleri ve öz-dikkat mekanizması için bir "fren mesafesi" oluşturularak son hecelerin kesilmesi (*truncation*) önlenir.
3. **Başlangıç Tamponu (Leading Silence Buffer, 100 ms):**
   - Sesin başlangıcına 100 ms sessizlik eklenerek ilk patlamalı ünsüzlerin (*p, t, k, ç*) ani kesilmeye uğraması engellenir.
4. **Projektörde Çerçeve Koruma (`projector_padding_mode: "pad"`):**
   - Baseline (`t0`) modelindeki `"discard"` modu, 5'in katı olmayan artık kareleri ($T \pmod 5$, 20-80 ms) çöpe atıyordu. `"pad"` modu ile artık kareler sıfır dolgu yapılarak son hecelerin tamamı LLM'e aktarılır.
5. **Erken Bitiş Cezası (`length_penalty: 1.2`):**
   - Dil modelinin erken `<eos>` üretmesini engelleyerek cümlenin sonuna kadar transkribe edilmesini sağlar.

---

## 📊 Veri Seti: ISSAI Turkish Speech Corpus (ISSAI_TSC_218)

* **Hugging Face:** [`issai/Turkish_Speech_Corpus`](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
* **Arşiv Dosyası:** `ISSAI_TSC_218.tar.gz` (~21.4 GB)
* **Ses Süresi:** ~218.2 saat (186.171 kayıt, 16 kHz WAV)
* **Lisans:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 🏗️ Proje Yapısı

```
tr/tsc/t1-preprocess/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # Donanıma göre optimize DRC ve pad konfigürasyonu
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle anlık test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # FrameDownsampler, LinearProjector, MLPProjector
│   │   ├── speech_encoder.py           # SpeechEncoderWrapper (Whisper, MMS, HuBERT)
│   │   └── slam_model.py               # SLAM_ASR ana modeli (length_penalty desteğiyle)
│   ├── data/
│   │   ├── audio_processor.py          # Dinamik Aralık Sıkıştırma (DRC) ve kuyruk tamponu
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
├── tests/                              # Birim testler (test_audio_processor dahil 36 test)
├── train.py                            # Model eğitim betiği
├── evaluate.py                         # Model değerlendirme betiği (length_penalty ve DRC ile)
├── inference.py                        # Tekil ses transkripsiyon betiği (length_penalty ve DRC ile)
├── run_tests.py                        # Test koşucu
├── slam-asr-tsc.zip                    # Taşınabilir kod arşivi
└── slam_asr_colab_tr_tsc_t1_preprocess.ipynb # Google Colab tek tıkla çalıştırma defteri
```
