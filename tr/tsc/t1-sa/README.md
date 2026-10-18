# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, **ISSAI Turkish Speech Corpus (ISSAI_TSC_218)** veri seti (~218 saatlik Türkçe konuşma verisi) ve modern Türkçe/çokdilli temel modeller (OpenAI Whisper + Qwen2.5 / Trendyol-LLM) kullanılarak geliştirilmiştir.

---

## 📖 Genel Bakış / Overview (t1-sa: Spectral Attention)

Bu klasör (`t1-sa`), akustik frekans kanalları ve harmonik formant yapıları arasındaki korelasyonu modellemek üzere **Spectral Attention (Spektral Dikkat)** mekanizmasının entegre edildiği deneysel daldır:
* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` — **DONDURULMUŞ (FROZEN)**.
* **Büyük Dil Modeli (LLM Backbone):** `Qwen/Qwen2.5-7B-Instruct` — **DONDURULMUŞ (FROZEN)**.
* **Projektör (`SpectralAttentionProjector`):**
  1. **Cross-Subband Spectral Self-Attention (16 Alt-Bant):** Akustik özellik boyutunu ($1280$) 16 spektral alt-banda (her biri 80 boyutlu) böler. Bu alt-bantlar arasında çok başlıklı self-attention kurarak temel frekans ($F_0$), üst armonikler ve ünlü formantları ($F_1-F_2$) arasındaki harmonik bağıntıları frekans kanalları boyunca modeller.
  2. **Dynamic Spectral Channel Gating (Squeeze-and-Excitation):** Zamansal havuzlama ve darboğaz katmanları ile frekans kanallarını dinamik olarak yeniden kalibre eder; belirgin formant frekanslarını öne çıkarırken arka plan ortam gürültüsü kanallarını baskılar.
  3. **Zamansal Sıkıştırma (Downsampling):** Spektral olarak arındırılmış $k=5$ ardışık çerçevenin birleştirilmesi (50 Hz $\to$ 10 Hz).
  4. **Lineer Projeksiyon:** Birleştirilmiş özellikleri LLM gömme boyutuna ($3584$) eşleme.

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
tr/tsc/t1-sa/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # Spectral Attention konfigürasyonu
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle anlık test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # SpectralSubBandAttention, SpectralChannelGating, SpectralAttentionProjector
│   │   ├── speech_encoder.py           # SpeechEncoderWrapper (Whisper, MMS, HuBERT)
│   │   └── slam_model.py               # SLAM_ASR ana modeli (Spectral Attention parametreleri)
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
├── tests/                              # Kapsamlı birim testleri (Spectral Attention dahil)
├── train.py                            # Model eğitim betiği
├── evaluate.py                         # Model değerlendirme betiği (WER/CER)
├── inference.py                        # Tekil ses transkripsiyon betiği
├── run_tests.py                        # Test koşucu
├── slam_asr_colab_tr_tsc_t1_sa.ipynb   # Colab eğitim notebook'u
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

## 🔬 Spectral Attention Hipotezi ve Tasarım Mantığı

Geleneksel zaman-odaklı self-attention mekanizmaları, yalnızca zaman eksenindeki ($T \times T$) çerçeve ilişkilerine bakar. Ancak insan sesinde:
1. **Harmonik ve Formant Korelasyonu:** Ünlü seslerin kimliğini belirleyen $F_1$ ve $F_2$ formantları ve $F_0$ armonikleri frekans spektrumu boyunca birbirinden bağımsız değildir.
2. **Kanal Bazlı Gürültü Arındırma:** Ortam gürültüleri spektrumun belli frekans bantlarında toplanırken, konuşma enerjisi belirli harmonik bantlarda yoğunlaşır.

`SpectralAttentionProjector`, özellik boyutunu 16 spektral alt-banda bölüp aralarında cross-subband dikkat kurarak ve ardından dinamik kanal kapılaması uygulayarak gürültüyü baskılar ve LLM'e giden akustik özelliklerin spektral netliğini artırır.
