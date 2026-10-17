# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu çalışma, **ISSAI Turkish Speech Corpus (ISSAI_TSC_218)** veri seti (~218 saatlik Türkçe konuşma verisi) ve modern Türkçe/çokdilli temel modeller (OpenAI Whisper + Qwen2.5 / Trendyol-LLM) kullanılarak geliştirilmiştir.

---

## 📖 Genel Bakış / Overview (t1-ma: Morpheme Attention)

Bu klasör (`t1-ma`), SLAM-ASR mimarisi üzerine sondan eklemeli (agglutinative) diller için özel olarak geliştirilen **Morpheme Attention (Morfem Dikkati)** mekanizmasının entegre edildiği deneysel daldır:
* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` — **DONDURULMUŞ (FROZEN)**.
* **Büyük Dil Modeli (LLM Backbone):** `Qwen/Qwen2.5-7B-Instruct` — **DONDURULMUŞ (FROZEN)**.
* **Projektör (`MorphemeAttentionProjector`):**
  1. **Multi-Scale Morphemic Convolutions (Çok Ölçekli Evrişim):** Farklı akustik sürelerdeki Türkçe morfem yapılarını yakalamak için paralel derinlemesine (depthwise) 1D evrişim katmanları:
     - Kısa çekim ekleri (~60 ms, çekirdek boyutu 3)
     - Standart yapım ve çekim ekleri (~100 ms, çekirdek boyutu 5)
     - Kökler ve birleşik morfem yapıları (~220 ms, çekirdek boyutu 11)
     - Dinamik kanal geçitlemesi ve artık (residual) bağlantı ile birleştirme.
  2. **Zamansal Sıkıştırma (Downsampling):** $k=5$ ardışık çerçevenin birleştirilmesi (50 Hz $\to$ 10 Hz).
  3. **Lineer Projeksiyon:** Birleştirilmiş özellikleri LLM gömme boyutuna ($3584$) eşleme.
  4. **Morpheme Prototype Cross-Attention (128 Yuva):** 128 adet öğrenilebilir Türkçe morfolojik kategori yuvası (isim/fiil kökleri, durum ekleri, iyelik, zaman/kip, şahıs ve yapım ekleri) ile çok başlıklı çapraz dikkat kurarak akustik kareleri morfolojik işlevsel rollere bağlar.
  5. **Inter-Morphemic Contextual Self-Attention (4 Başlıklı):** Morfem tokenları arasında dizi boyunca bağlamsal self-attention kurarak Türkçe ünlü uyumu ve kök-ek uyum kurallarını modeller.

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
tr/tsc/t1-ma/
├── config/
│   ├── hardware_adaptive_turkish.yaml  # Morpheme Attention konfigürasyonu
│   ├── colab_t4_turkish.yaml           # Colab T4 / RTX A4000 için 4-bit NF4 konfigürasyonu
│   ├── train_whisper_trendyol_turkish.yaml # Whisper + Trendyol-LLM
│   └── train_toy_debug_turkish.yaml    # Sentetik veriyle anlık test konfigürasyonu
├── slam_asr/
│   ├── models/
│   │   ├── projector.py                # MultiScaleMorphemeConv, MorphemePrototypeAttention, MorphemeAttentionProjector
│   │   ├── speech_encoder.py           # SpeechEncoderWrapper (Whisper, MMS, HuBERT)
│   │   └── slam_model.py               # SLAM_ASR ana modeli (Morpheme Attention parametreleri)
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
├── tests/                              # Kapsamlı birim testleri (Morpheme Attention dahil)
├── train.py                            # Model eğitim betiği
├── evaluate.py                         # Model değerlendirme betiği (WER/CER)
├── inference.py                        # Tekil ses transkripsiyon betiği
├── run_tests.py                        # Test koşucu
├── slam_asr_colab_tr_tsc_t1_ma.ipynb   # Colab eğitim notebook'u
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

## 🔬 Morpheme Attention (MA) Hipotezi ve Tasarım Mantığı

Türkçe gibi sondan eklemeli ve zengin çekim morfolojisine sahip dillerde standart SLAM-ASR'nin karşılaştığı temel zorluklar:
1. **Değişken Morfem Süreleri:** Ekler tek bir sesten oluşabileceği gibi (`-i`, `-e` ~60ms) birden fazla heceyi kapsayabilir (`-ebilmek`, `-cesine` ~200+ ms). Sabit tek ölçekli lineer katman bu farklılaşmayı kaçırabilir. Çok ölçekli evrişim katmanı (3, 5, 11) farklı akustik morfem sürelerini ayrıştırır.
2. **Morfolojik Rol Eşleme:** 128 adet öğrenilebilir prototip slotu, konuşma gösterimlerinin morfolojik işlevlerine göre kümelenmesine yardımcı olur.
3. **Ünlü Uyumu ve Morfotaktik Bağımlılıklar:** Kökün son seslisi ile sonraki eklerin seslileri arasındaki büyük ve küçük ünlü uyumları, Inter-Morphemic Self-Attention katmanı sayesinde dizi boyunca etkin biçimde modellenir.
