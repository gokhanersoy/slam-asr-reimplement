# Turkish SLAM-ASR: Doğrusal Projektör + PEFT LoRA + Kuyruk Veri Artırma Ablasyonu (t1-tail-aug-lora)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu proje (`t1-tail-aug-lora`), tezinizdeki ablasyon çalışmalarında **PEFT LoRA (Dil Modeli İnce Ayarı)** ile **Hedefe Yönelik Kuyruk Veri Artırmasını (Targeted Tail Data Augmentation)** bir araya getiren ve bu iki yaklaşımın birleştiğinde sunduğu **sinerjiyi** ölçen kritik bir deneydir.

---

## 🎯 $2 \times 2$ Faktöriyel Ablasyon Matrisi (Factorial Ablation Matrix)

Dil modeli adaptasyonu (LoRA) ile hedefli kuyruk veri artırmasının etkilerini ayrıştırmak için kurulan deney matrisi:

| Deney Kodu | Projektör Mimarisi | LLM Durumu | Veri Artırma | ISSAI TSC (WER) | Mozilla CV-TR (WER) | Bilimsel Rolü |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`t0` (Temel / Baseline)** | Standart Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Yok (Standart) | %12.98 | %14.85 | Orijinal SLAM-ASR referans modeli |
| **`t1-lora`** | Standart Doğrusal ($k=5$) | PEFT LoRA ($r=16, \alpha=32$) | Yok (Standart) | *Eğitilecek* | *Eğitilecek* | Doğrusal projektörde LoRA'nın saf etkisi |
| **`t1-tail-aug`** | Standart Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Kuyruk Hızlandırma + Sönümleme | *Eğitilecek* | *Eğitilecek* | Dondurulmuş modelde kuyruk artırmasının saf etkisi |
| **`t1-tail-aug-lora` (Bu Proje)** | **Standart Doğrusal ($k=5$)** | **PEFT LoRA ($r=16, \alpha=32$)** | **Kuyruk Hızlandırma + Sönümleme** | *Eğitilecek* | *Eğitilecek* | **LoRA + Kuyruk Artırması Sinerjisi** |
| **`t4-ng-ta-gormez-lora`** | Görmez Hibrit (N-gram + Attn + BiGRU) | PEFT LoRA ($r=16, \alpha=32$) | Yok (Standart) | **%11.53** | **%13.16** | Tüm zamanların SOTA rekoru |

### Bu Deneyin Cevaplayacağı Bilimsel Sorular:
1. **LoRA varken Kuyruk Artırması İlave Kazanç Sağlar mı?**  
   $\text{WER}(t1\text{-lora}) \leftrightarrow \text{WER}(t1\text{-tail-aug-lora})$: LoRA ile LLM adapte edilirken, eğitim verisinin son bölümünü hızlandırıp sönümlendirmek cümle sonu harf/ek yutma hatalarını LoRA'nın tek başına çözebildiğinden daha ileri seviyede düzeltir mi?
2. **Kuyruk Artırmalı Doğrusal Model, Görmez Projektöre Ne Kadar Yaklaşır?**  
   $\text{WER}(t1\text{-tail-aug-lora}) \leftrightarrow \text{WER}(t4\text{-ng-ta-gormez-lora})$: Akustik seviyedeki veri artırması, mimari karmaşıklık eklemeden doğrusal bir modelin SOTA performansına yaklaşmasını sağlayabilir mi?

---

## 🏗️ Mimari ve Veri Bileşenleri

1. **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-large-v3` (1280 dim, [DONDURULMUŞ / FROZEN]).
   - Ham konuşma sinyalinden 50 Hz log-mel filtre bankası öznitelikleri çıkarır.
2. **Standart Doğrusal Projektör (Linear Projector):**
   - $k=5$ zaman boyutunda ardışık kare birleştirme (temporal downsampling, 50 Hz $\to$ 10 Hz, $1280 \times 5 = 6400 \to 3584$ dim).
3. **Büyük Dil Modeli ve LoRA Adaptasyonu (LLM Backbone + PEFT LoRA):**
   - `Qwen/Qwen2.5-7B-Instruct` temel modeli (4-bit NF4 kuantizasyon).
   - Dikkat projeksiyonlarına (`q_proj`, `v_proj`, `k_proj`, `o_proj`) $r=16, \alpha=32, \text{dropout}=0.05$ LoRA adaptörleri eklenir.
   - Base LLM dondurulurken sadece projektör (~22MB) ve LoRA adaptörleri (~50MB) eğitilir.
4. **Hedefe Yönelik Kuyruk Veri Artırma Boru Hattı (`tail_augmentations.py`):**
   - **`TailSpeedPerturbation`:** Cümlenin son $\%20 - \%35$'lik dilimi $1.15\times - 1.35\times$ hızlandırılır (40 ms linear cross-fade).
   - **`TailEnergyDecay`:** Nefes tükenmesini ve fısıltılı konuşmayı simüle etmek için kuyruk genliğine $-3\text{ dB}$ ila $-8\text{ dB}$ sönümleme uygulanır.
   - **`TailSpecMasking`:** Yalnızca aktif konuşma kuyruğunun Mel spektrogram karelerinde zaman ve frekans maskelemesi yoğunlaştırılır.

---

## 💾 Kontrol Noktaları ve Ağırlık Yönetimi (Checkpoints)

SLAM-ASR eğitiminde 15 GB'lık ana LLM ağırlıkları diske tekrar yazılmaz:
* `projector.pt`: Eğitilen doğrusal projektör ağırlıkları (~22 MB)
* `lora_adapter/`: Hugging Face PEFT LoRA ağırlıkları (`adapter_model.safetensors`, `adapter_config.json`) (~50 MB)
* `config.pt`: SLAM-ASR konfigürasyon nesnesi

`evaluate.py` ve `inference.py` kontrol noktası dizininde `lora_adapter` bulunduğunda adaptörü otomatik olarak yükler.

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
