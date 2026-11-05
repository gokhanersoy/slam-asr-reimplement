# Turkish SLAM-ASR: Görmez Hibrit Projektör + PEFT LoRA + Kuyruk Veri Artırma + Zemberek Dil Modeli (t6-ng-ta-ctx-tail-aug-lora-lm)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu proje (`t6-ng-ta-ctx-tail-aug-lora-lm`), teziniz boyunca geliştirilen şampiyon model `t5-ng-ta-ctx-tail-aug-lora` üzerine, Görmez'in (2024) *"Customized deep learning based Turkish automatic speech recognition system supported by language model"* çalışmasında kullandığı **Zemberek Morfolojik Dil Modeli (Language Model - `lm`)** desteğini entegre eden ileri araştırma modelidir.

---

## 🎯 Temel Güçlerin Birleşimi (The 4 Pillars of t6)

1. **Mimari Güç (`ng-ta-ctx`):** NeMo 1D derinlemesine ayrılabilir kapılı evrişimler ile 4 başlı Öz-Dikkat (Multi-Head Attention) kollarını dinamik kapılama ile birleştiren bağlam duyarlı hibrit projektör.
2. **Dil Modeli Adaptasyon Gücü (`lora`):** 8B parametreli LLaMA-3 / Qwen2.5 tabanlı üretken LLM omurgasına uygulanan PEFT LoRA ($r=16, \alpha=32$) ince ayarı.
3. **Akustik Dayanıklılık Gücü (`tail-aug`):** Türkçedeki çekim eklerinin kelime sonuna yığılması nedeniyle konuşma sonundaki enerji düşüşlerini ve kesilmelerini önleyen hedefli kuyruk veri artırma boru hattı (%3.15 CER rekoru).
4. **Morfolojik Dil Modeli Gücü (`lm` - Zemberek):** Türkçenin büyük/küçük ünlü uyumları, ünsüz yumuşaması/benzeşmesi kurallarını denetleyen morfolojik çözümleyici ve n-gram dil modeli ile ASR hipotezlerinin yeniden puanlanması (rescoring) ve düzeltilmesi.

---

## 📊 Evrimsel Kıyaslama Matrisi (Evolutionary Benchmark Matrix)

| Deney Kodu | Projektör Mimarisi | LLM Durumu | Veri Artırma / Ön İşleme | Dil Modeli / Son İşleme | ISSAI (WER) | ISSAI (CER) | CV (WER) | CV (CER) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`t0` (Baseline)** | Standart Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Yok | Yok | %13.22 | %3.88 | %14.87 | %4.08 |
| **`t1-ng`** | NeMo Kapılı Conv1D | Dondurulmuş (Frozen) | Yok | Yok | %12.49 | %3.68 | %14.33 | %3.76 |
| **`t2-ng-ta`** | Kapılı Conv1D + Zaman | Dondurulmuş (Frozen) | Yok | Yok | %12.36 | %3.55 | %14.04 | %4.05 |
| **`t3-ng-ta-ctx`** | Çift Kollu Bağlam Duyarlı | Dondurulmuş (Frozen) | Yok | Yok | %12.27 | %3.46 | %13.91 | %3.99 |
| **`t1-lora`** | Standart Doğrusal ($k=5$) | PEFT LoRA ($r=16, \alpha=32$) | Yok | Yok | %11.78 | %3.40 | %13.56 | %3.52 |
| **`t4-ng-ta-ctx-lora`** | Çift Kollu Bağlam Duyarlı | PEFT LoRA ($r=16, \alpha=32$) | Yok | Yok | %11.53 | %3.27 | %13.16 | %3.34 |
| **`t4-ng-ta-ctx-postcorr`**| Çift Kollu Bağlam Duyarlı | Dondurulmuş (Frozen) | Yok | Arslan Sözlük Düzeltmesi | %13.60 | %3.80 | %16.99 | %4.99 |
| **`t5-ng-ta-ctx-tail-aug-lora`** | **Çift Kollu Bağlam Duyarlı** | **PEFT LoRA ($r=16, \alpha=32$)** | **Kuyruk Veri Artırma** | **Yok (Uçtan Uca SOTA)** | **%11.43** | **%3.21** | **%12.75** | **%3.17** |
| **`t6-ng-ta-ctx-tail-aug-lora-lm` (Bu Model)** | **Çift Kollu Bağlam Duyarlı** | **PEFT LoRA ($r=16, \alpha=32$)** | **Kuyruk Veri Artırma** | **Zemberek LM (Görmez 2024)** | **Hedefleme** | **Hedefleme** | **Hedefleme** | **Hedefleme** |

---

## 🏗️ Zemberek Dil Modeli (`slam_asr/utils/zemberek_lm.py`)

Zemberek modülü iki ana çalışma motoruna sahiptir:
1. **Yerel Python Motoru (Sıfır Bağımlılık / Zero-Dependency):**
   - Türkçenin büyük ve küçük ünlü uyumlarını, ek zincirlerini ve kök türetimlerini denetler.
   - Witten-Bell ve Jelinek-Mercer yumuşatmalı Trigram / Bigram dil modeli ile cümle olasılıklarını ($-\log P$) ve şaşkınlık (Perplexity - PPL) değerlerini hesaplar.
   - Her ortamda (CPU, GPU, test suite) harici Java kurulumu gerektirmeden %100 taşınabilir çalışır.
2. **Yerel Java Zemberek Motoru (JPype):**
   - Ortamda `jpype1` ve `zemberek-full.jar` mevcut olduğunda Java JVM üzerinden resmi `TurkishMorphology` ve `TurkishSpellChecker` sınıflarına doğrudan bağlanır.

---

## 🚀 Hızlı Başlangıç

### 1. Testleri Çalıştırma
```bash
python run_tests.py
```

### 2. Eğitimi Başlatma
```bash
python train.py --config config/hardware_adaptive_turkish.yaml
```

### 3. Zemberek LM ile Değerlendirme (WER / CER)
```bash
python evaluate.py \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint outputs/slam_asr_turkish_adaptive_t6/best_checkpoint \
    --manifest data/issai_tsc/test.jsonl \
    --use_zemberek_lm \
    --lm_weight 0.3 \
    --beam_size 4
```

`evaluate.py` hem ham ASR (Raw ASR) hem de Zemberek LM destekli sonuçları ve aralarındaki net WER/CER kazancını ekrana yazdırır.

### 4. Tekli Ses Dosyası Transkripsiyonu (Inference)
```bash
python inference.py \
    --audio data/toy_audio_tr/tr_toy_001.wav \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint outputs/slam_asr_turkish_adaptive_t6/best_checkpoint \
    --use_zemberek_lm
```
