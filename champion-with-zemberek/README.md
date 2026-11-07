# 🏆 Turkish SLAM-ASR: Şampiyon Model (`t5`) + Zemberek Dil Modeli (`champion-with-zemberek`)

Bu klasör, tez ve makale çalışmalarımızda **en yüksek başarıma ulaşan şampiyon modelimiz `t5-ng-ta-ctx-tail-aug-lora`** üzerine, literatürde Görmez'in [-@gormez2024customized] kullandığı **Zemberek Morfolojik Dil Modeli (Language Model - LM)** desteğini entegre eden özel çalışma paketidir.

---

## 🎯 Temel İlke: Sıfır Eğitim Adımı (Zero-Training Inference & Evaluation)

Bu paketin en önemli özelliği, **bütün eğitim adımlarını baştan çalıştırma zorunluluğunu tamamen ortadan kaldırmasıdır**:
* Önceden eğitilmiş olan `t5` model ağırlıkları (`best_checkpoint`: `projector.pt` ve `lora_adapter`) doğrudan Google Drive veya yerel diskten yüklenir.
* Model baştan eğitilmez; doğrudan çıkarım (inference) ve test seti değerlendirmesi (evaluation) yapılır.
* Eğer daha önce üretilmiş bir tahmin dosyası (`eval_results.jsonl`) mevcutsa, GPU ile saatlerce çıkarım yapmaya dahi gerek kalmadan **30 saniye içinde** Zemberek LM ile yeniden puanlama (rescoring) ve morfolojik düzeltme uygulanarak nihai WER ve CER değişimleri hesaplanabilir.

---

## 🔬 Mimarinin 4 Temel Gücü

1. **Mimari Güç (`ng-ta-ctx`):** NeMo 1D derinlemesine ayrılabilir kapılı evrişimler ile 4 başlı Öz-Dikkat (Multi-Head Attention) kollarını dinamik kapılama ile birleştiren çift kollu bağlam duyarlı hibrit projektör.
2. **Dil Modeli Adaptasyon Gücü (`lora`):** 8 milyar parametreli LLaMA-3 / Qwen2.5 tabanlı üretken LLM omurgasına uygulanan PEFT LoRA ($r=16, \alpha=32$) ince ayarı.
3. **Akustik Dayanıklılık Gücü (`tail-aug`):** Türkçedeki çekim eklerinin kelime sonuna yığılması nedeniyle konuşma sonundaki enerji düşüşlerini ve kesilmelerini önleyen hedefli kuyruk veri artırma boru hattı.
4. **Morfolojik Dil Modeli Gücü (`lm` - Zemberek):** Türkçenin büyük/küçük ünlü uyumları, ünsüz yumuşaması/benzeşmesi kurallarını denetleyen kural tabanlı morfolojik çözümleyici ve n-gram dil modeli ile ASR hipotezlerinin yeniden puanlanması ve düzeltilmesi (Görmez, 2024 LM desteği).

---

## 📁 Klasör Yapısı

```
champion-with-zemberek/
├── slam_asr_champion_with_zemberek.ipynb  # 🚀 Ana Google Colab / Jupyter Notebook (Sıfır Eğitim)
├── README.md                              # 📖 Bu dokümantasyon dosyası
├── evaluate.py                            # 🎙️ Uçtan uca test ve Zemberek LM değerlendirme scripti
├── inference.py                           # 🔊 Tekil ses dosyası transkripsiyonu ve LM düzeltmesi
├── requirements.txt                       # 📦 Gerekli Python bağımlılıkları
├── pyproject.toml                         # ⚙️ Paket yapılandırması
├── run_tests.py                           # 🧪 Birim test çalıştırıcısı
├── config/
│   ├── colab_t4_turkish.yaml             # Google Colab T4 (16GB) yapılandırması
│   └── hardware_adaptive_turkish.yaml    # Donanımı otomatik algılayan yapılandırma
├── data/
│   ├── sample_train_tr.jsonl             # Hızlı doğrulama için örnek veri manifestosu
│   ├── sample_val_tr.jsonl               # Hızlı doğrulama için örnek doğrulama manifestosu
│   └── toy_audio_tr/                     # Örnek test sesleri (.wav)
├── scripts/
│   ├── apply_zemberek_lm.py              # ⚡ Mevcut eval_results.jsonl'e 30 saniyede LM uygulama
│   └── build_zemberek_lm.py              # 📊 Metin verisinden n-gram LM tablosu çıkarma
├── slam_asr/
│   ├── models/                           # Whisper Encoder, Görmez Hibrit Projektör, SLAM_ASR
│   ├── data/                             # Ses işleme ve veri yükleyiciler
│   └── utils/
│       ├── zemberek_lm.py                # 🧠 Zemberek Morfolojik Dil Modeli & N-Gram Rescorer
│       ├── metrics.py                    # Türkçe WER / CER ve normalizasyon
│       └── device_utils.py               # Donanım profilleyicisi
└── tests/
    ├── test_zemberek_lm.py               # Zemberek LM birim testleri (7 test)
    ├── test_projector.py                 # Projektör mimarisi testleri
    └── test_metrics_turkish.py           # Türkçe metrik testleri
```

---

## 🚀 Kullanım Kılavuzu

### 1. Google Colab Notebook ile Çalıştırma (En Pratik Yol)
`slam_asr_champion_with_zemberek.ipynb` dosyasını Google Colab'a yükleyin. Notebook adım adım şu sırayı takip eder:
1. **GPU & Drive:** GPU'yu doğrular ve Google Drive'ınızı bağlar.
2. **Bağımlılıklar:** Gerekli kütüphaneleri yükler.
3. **Checkpoint Tespiti:** Drive'daki `t5-ng-ta-ctx-tail-aug-lora` ağırlıklarını (`projector.pt` ve `lora_adapter`) otomatik bulur.
4. **Zemberek LM Başlatma:** Morfolojik kuralları ve n-gram modelini saniyeler içinde belleğe alır.
5. **⚡ HIZLI YOL:** Eğer daha önce `eval_results.jsonl` ürettiyseniz, modeli baştan çalıştırmadan 30 saniyede Zemberek LM uygulayıp karşılaştırma tablosunu çıkarır.
6. **🎙️ Uçtan Uca Değerlendirme:** Test seti üzerinde `evaluate.py` çalıştırarak Ham ASR vs Zemberek LM WER/CER skorlarını hesaplar.
7. **🔊 İnteraktif Çıkarım:** Seçilen bir sesi dinletir ve iki modelin çıktısını kıyaslar.
8. **🎤 Canlı Mikrofon:** Tarayıcıdan ses kaydedip anında tanır.
9. **💾 Drive Yedekleme:** Sonuçları Drive'a otomatik kaydeder.

---

### 2. Komut Satırından Çalıştırma (CLI)

#### A. Mevcut Tahmin Dosyasına Zemberek LM Uygulama (30 Saniye):
```bash
python scripts/apply_zemberek_lm.py \
    --input eval_results_t5.jsonl \
    --output eval_results_champion_zemberek_lm.jsonl \
    --train_manifest data/issai_tsc/train.jsonl
```

#### B. Uçtan Uca Test Seti Değerlendirmesi:
```bash
python evaluate.py \
    --config config/colab_t4_turkish.yaml \
    --checkpoint /path/to/t5-ng-ta-ctx-tail-aug-lora/best_checkpoint \
    --manifest data/issai_tsc/test.jsonl \
    --output_file eval_results_champion_zemberek.jsonl \
    --use_zemberek_lm \
    --beam_size 4
```

#### C. Tekil Ses Dosyası Transkripsiyonu:
```bash
python inference.py \
    --audio data/toy_audio_tr/tr_toy_001.wav \
    --config config/colab_t4_turkish.yaml \
    --checkpoint /path/to/t5-ng-ta-ctx-tail-aug-lora/best_checkpoint \
    --use_zemberek_lm
```

---

## 🧪 Birim Testleri Çalıştırma

Tüm Zemberek LM fonksiyonlarının, büyük ünlü uyumu kurallarının, n-gram puanlayıcılarının ve morfolojik analizcilerin doğruluğunu teyit etmek için:
```bash
python run_tests.py
```
*(Tüm testler milisaniyeler içinde başarıyla tamamlanmaktadır.)*
