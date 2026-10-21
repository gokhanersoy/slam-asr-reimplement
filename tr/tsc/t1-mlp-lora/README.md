# Turkish SLAM-ASR: 2 Katmanlı MLP Projektör + PEFT LoRA Ablasyon Çalışması (t1-mlp-lora)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu proje (`t1-mlp-lora`), tez çalışmasında **Projektör Karmaşıklığı (Doğrusal vs 2 Katmanlı MLP vs Görmez Çok Kollu Hibrit)** ile **Dil Modeli İnce Ayarı (Dondurulmuş vs PEFT LoRA)** eksenlerini karşılaştıran faktöriyel ablasyon ağının kritik bir deneyidir.

---

## 🎯 Ablasyon Matrisi (Factorial Ablation Matrix)

Mimari tasarım ile PEFT LoRA etkileşimini tam olarak haritalamak için oluşturulan deney matrisi:

| Deney Kodu | Projektör Mimarisi | LLM Durumu | ISSAI TSC (WER) | Mozilla CV-TR (WER) | Bilimsel Rolü |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`t0` (Temel / Baseline)** | Standart Doğrusal ($k=5$) | Dondurulmuş (Frozen) | %12.98 | %14.85 | Orijinal SLAM-ASR referans modeli |
| **`t1-lora`** | Standart Doğrusal ($k=5$) | PEFT LoRA ($r=16, \alpha=32$) | *Eğitilecek* | *Eğitilecek* | Doğrusal projektörde LoRA'nın saf etkisi |
| **`t1-mlp-lora` (Bu Proje)** | **2 Katmanlı MLP (ReLU, $k=5$)** | **PEFT LoRA ($r=16, \alpha=32$)** | *Eğitilecek* | *Eğitilecek* | **Doğrusal-olmayan standart MLP + LoRA** |
| **`t3-ng-ta-gormez`** | Görmez Hibrit (N-gram + Attn + BiGRU) | Dondurulmuş (Frozen) | %12.27 | %14.02 | Saf mimari iyileştirme kazancı ($\Delta -0.71\%$) |
| **`t4-ng-ta-gormez-lora`** | Görmez Hibrit (N-gram + Attn + BiGRU) | PEFT LoRA ($r=16, \alpha=32$) | **%11.53** | **%13.16** | Tüm zamanların SOTA rekoru ($\Delta -1.45\%$) |

### Bu Deneyin Cevaplayacağı 2 Temel Soru:
1. **LoRA varken Doğrusal Projektör ile Standart MLP Arasındaki Fark:** `t1-lora` $\leftrightarrow$ `t1-mlp-lora` karşılaştırması, LLM adapte edilirken projektörde 2 katmanlı standart bir MLP kullanmanın doğrusal projektöre göre anlamlı bir avantaj sağlayıp sağlamadığını gösterir.
2. **Görmez Hibrit Mimarisinin Standart MLP'ye Üstünlüğü:** `t1-mlp-lora` $\leftrightarrow$ `t4-ng-ta-gormez-lora` karşılaştırması, `t4`'teki SOTA başarısının sadece "doğrusal-olmayan" bir projektörden mi yoksa Görmez mimarisindeki akustik n-gram, zamansal dikkat havuzlama ve çift yönlü GRU akustik hafıza bileşenlerinden mi kaynaklandığını kesin olarak kanıtlar.

---

## 🏗️ Mimari ve Bileşenler (Architecture)

1. **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` (1280 dim, [DONDURULMUŞ / FROZEN]).
   - Ham konuşma sinyalinden 50 Hz log-mel filtre bankası öznitelikleri çıkarır.
2. **2 Katmanlı MLP Projektör (2-Layer MLP Projector):**
   - $k=5$ zaman boyutunda ardışık kare birleştirme (temporal downsampling, 50 Hz $\to$ 10 Hz, $1280 \times 5 = 6400$ dim).
   - $\text{Linear}(6400, 2048) \to \text{ReLU}() \to \text{Linear}(2048, 3584)$ — [EĞİTİLEBİLİR] (~40 MB).
3. **Büyük Dil Modeli ve LoRA Adaptasyonu (LLM Backbone + PEFT LoRA):**
   - `Qwen/Qwen2.5-7B-Instruct` temel modeli (4-bit NF4 kuantizasyon).
   - Dikkat projeksiyonlarına (`q_proj`, `v_proj`, `k_proj`, `o_proj`) $r=16, \alpha=32, \text{dropout}=0.05$ LoRA adaptörleri eklenir.
   - Base LLM dondurulurken sadece MLP projektör (~40MB) ve LoRA adaptörleri (~50MB) eğitilir.

---

## 💾 Kontrol Noktaları ve Ağırlık Yönetimi (Checkpoints)

SLAM-ASR eğitiminde 15 GB'lık ana LLM ağırlıkları asla diske tekrar yazılmaz:
* `projector.pt`: Eğitilen 2 katmanlı MLP projektör ağırlıkları (~40 MB)
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
