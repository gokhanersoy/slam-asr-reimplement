# Turkish SLAM-ASR: Doğrusal Projektör + PEFT LoRA Ablasyon Çalışması (t1-lora)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu proje (`t1-lora`), tez çalışmasında **LoRA adaptasyonunun sağladığı saf kazancı** doğrusal baz mimari (`t0`) üzerinde izole etmek amacıyla tasarlanmış **$2 \times 2$ Faktöriyel Ablasyon Analizinin (Factorial Ablation Study)** kritik bileşenidir.

---

## 🎯 $2 \times 2$ Faktöriyel Ablasyon Matrisi (Factorial Ablation Matrix)

Mimari yenilikler (Görmez Çok Kollu Hibrit Projektör) ile parametre verimli dil modeli ince ayarının (PEFT LoRA) etkilerini birbirinden ayrıştırmak için kurulan deney matrisi:

| Deney Kodu | Projektör Mimarisi | LLM Durumu | ISSAI TSC (WER) | Mozilla CV-TR (WER) | Açıklama |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`t0` (Temel / Baseline)** | Standart Doğrusal ($k=5$) | Dondurulmuş (Frozen) | %12.98 | %14.85 | Orijinal SLAM-ASR referans modeli |
| **`t3-ng-ta-gormez`** | Görmez Hibrit (N-gram + Attn + BiGRU) | Dondurulmuş (Frozen) | %12.27 | %14.02 | Saf mimari iyileştirme kazancı ($\Delta -0.71\%$) |
| **`t1-lora` (Bu Proje)** | **Standart Doğrusal ($k=5$)** | **PEFT LoRA ($r=16, \alpha=32$)** | *Eğitilecek* | *Eğitilecek* | **Doğrusal projektörde LoRA'nın saf etkisi** |
| **`t4-ng-ta-gormez-lora`** | Görmez Hibrit (N-gram + Attn + BiGRU) | PEFT LoRA ($r=16, \alpha=32$) | **%11.53** | **%13.16** | Tüm zamanların SOTA rekoru ($\Delta -1.45\%$) |

### Bu Deneyin Cevaplayacağı Bilimsel Sorular:
1. **LoRA'nın Doğrusal Projektöre Katkısı:** `t0` (%12.98) $\to$ `t1-lora` arasındaki fark, mimari değiştirmeden sadece LoRA eklemenin sağladığı net kazancı ortaya koyar.
2. **Görmez Hibrit Projektörün LoRA Üzerine Katkısı:** `t1-lora` $\to$ `t4-ng-ta-gormez-lora` (%11.53) arasındaki fark, LoRA varken bile Görmez hibrit projektörünün ne kadar ek kazanım sağladığını kanıtlar.

---

## 🏗️ Mimari ve Bileşenler (Architecture)

1. **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` (1280 dim, [DONDURULMUŞ / FROZEN]).
   - 50 Hz log-mel filtre bankası öznitelikleri çıkarır.
2. **Standart Doğrusal Projektör (Linear Projector):**
   - $k=5$ zaman boyutunda ardışık kare birleştirme (temporal downsampling, 50 Hz $\to$ 10 Hz, $1280 \times 5 = 6400$ dim).
   - Tek bir doğrusal projeksiyon katmanı ($6400 \to 3584$ dim) — [EĞİTİLEBİLİR].
3. **Büyük Dil Modeli ve LoRA Adaptasyonu (LLM Backbone + PEFT LoRA):**
   - `Qwen/Qwen2.5-7B-Instruct` temel modeli (4-bit NF4 kuantizasyon).
   - Dikkat projeksiyonlarına (`q_proj`, `v_proj`, `k_proj`, `o_proj`) $r=16, \alpha=32, \text{dropout}=0.05$ LoRA adaptörleri eklenir.
   - Base LLM dondurulurken sadece projektör (~22MB) ve LoRA adaptörleri (~50MB) eğitilir.

---

## 💾 Kontrol Noktaları ve Ağırlık Yönetimi (Checkpoints)

SLAM-ASR eğitiminde 15 GB'lık ana LLM ağırlıkları asla diske tekrar yazılmaz:
* `projector.pt`: Eğitilen doğrusal projektör ağırlıkları (~22 MB)
* `lora_adapter/`: Hugging Face PEFT LoRA ağırlıkları (`adapter_model.safetensors`, `adapter_config.json`) (~50 MB)
* `config.pt`: SLAM-ASR konfigürasyon nesnesi

`evaluate.py` ve `inference.py` kontrol noktası dizininde `lora_adapter` bulunduğunda adaptörü otomatik olarak yükler.

---

## 📊 Veri Seti: ISSAI Turkish Speech Corpus (ISSAI_TSC_218)

* **Hugging Face:** [`issai/Turkish_Speech_Corpus`](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
* **Arşiv:** `ISSAI_TSC_218.tar.gz` (~21.4 GB)
* **Ses Süresi:** ~218.2 saat (186.171 ses kaydı, 16 kHz WAV)
* **Lisans:** CC BY 4.0

### Veri Seti Hazırlığı
```bash
python scripts/prepare_issai_tsc.py --download --output_dir data/issai_tsc
```

---

## 🚀 Hızlı Başlangıç

### 1. Testleri Koşturma
```bash
python run_tests.py
```

### 2. Eğitimi Başlatma
* **Donanıma göre otomatik optimize edilen konfigürasyon ile:**
  ```bash
  python train.py --config config/hardware_adaptive_turkish.yaml
  ```
* **Colab T4 veya RTX A4000 üzerinde:**
  ```bash
  python train.py --config config/colab_t4_turkish.yaml
  ```

### 3. Değerlendirme (WER / CER Hesabı)
```bash
python evaluate.py \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint outputs/slam_asr_turkish_adaptive/best_checkpoint \
    --manifest data/issai_tsc/test.jsonl \
    --language tr \
    --beam_size 4
```

### 4. Ses Dosyasını Yazıya Dökme (Inference)
```bash
python inference.py \
    --audio /path/to/audio.wav \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint outputs/slam_asr_turkish_adaptive/best_checkpoint \
    --prompt "Konuşmayı metne dönüştürün."
```
