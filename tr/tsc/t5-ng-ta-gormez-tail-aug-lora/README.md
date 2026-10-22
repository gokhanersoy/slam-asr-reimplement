# Turkish SLAM-ASR: Görmez Hibrit Projektör + PEFT LoRA + Kuyruk Veri Artırma (t5-ng-ta-gormez-tail-aug-lora)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu proje (`t5-ng-ta-gormez-tail-aug-lora`), teziniz boyunca bağımsız olarak doğrulanmış ve rekor kırmış **en başarılı 3 temel gücü tek bir çatı altında birleştiren nihai zirve modeldir (The Capstone Model)**:

1. **Mimari Güç:** Görmez Çok Kollu Hibrit Projektör (`t4` ile **%11.53 WER** tüm zamanlar kelime hatası rekoru).
2. **Dil Modeli Adaptasyon Gücü:** Qwen2.5-7B üzerine entegre edilen PEFT LoRA ($r=16, \alpha=32$).
3. **Akustik Dayanıklılık Gücü:** Hedefe Yönelik Kuyruk Veri Artırması (`t1-tail-aug-lora` ile **%3.15 CER** tüm zamanlar karakter hatası rekoru).

---

## 🎯 Evrimsel Ablasyon ve Karşılaştırma Matrisi (Evolutionary Benchmark Matrix)

| Deney Kodu | Projektör Mimarisi | LLM Durumu | Veri Artırma / Ön İşleme | ISSAI TSC (WER) | ISSAI TSC (CER) | En İyi Val Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`t0` (Temel / Baseline)** | Standart Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Yok | %12.98 | %3.72 | 0.3015 |
| **`t1-preprocess`** | Standart Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Kuyruk Koruyucu + DRC | %13.29 | %3.86 | 0.3040 |
| **`t1-feature-aug`** | Standart Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Genel Hız (0.9-1.1x) + SpecAug | %13.17 | %3.82 | 0.2980 |
| **`t1-tail-aug`** | Standart Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Kuyruk Hızlandırma + Sönümleme | %13.17 | %3.76 | 0.2935 |
| **`t3-ng-ta-gormez`** | Görmez Hibrit (N-gram+Attn+BiGRU) | Dondurulmuş (Frozen) | Yok | %12.27 | %3.56 | 0.2850 |
| **`t1-mlp-lora`** | 2 Katmanlı MLP (ReLU) | PEFT LoRA ($r=16, \alpha=32$) | Yok | %11.99 | %3.35 | 0.2540 |
| **`t1-lora`** | Standart Doğrusal ($k=5$) | PEFT LoRA ($r=16, \alpha=32$) | Yok | %11.78 | %3.40 | 0.2526 |
| **`t1-tail-aug-lora`** | Standart Doğrusal ($k=5$) | PEFT LoRA ($r=16, \alpha=32$) | Kuyruk Hızlandırma + Sönümleme | %11.58 | **%3.15 🏆** | 0.2456 |
| **`t4-ng-ta-gormez-lora`** | Görmez Hibrit (N-gram+Attn+BiGRU) | PEFT LoRA ($r=16, \alpha=32$) | Yok | **%11.53 🏆** | %3.27 | 0.2480 |
| **`t5-ng-ta-gormez-tail-aug-lora` (Bu Model)** | **Görmez Hibrit** | **PEFT LoRA ($r=16, \alpha=32$)** | **Kuyruk Hızlandırma + Sönümleme** | **Hedef: < %11.30** | **Hedef: < %3.05** | **Zirve Hedefi** |

---

## 🏗️ Mimari ve Veri Bileşenleri Detayı

### 1. Akustik Kodlayıcı (Speech Encoder)
* `openai/whisper-large-v3` (1280 dim, [DONDURULMUŞ / FROZEN]). Ham konuşma sinyalinden 50 Hz log-mel filtre bankası öznitelikleri çıkarır.

### 2. Görmez Hiyerarşik Çok Kollu Hibrit Projektör (`GormezHybridAttentionProjector`)
* **Stage 1 (Akustik N-Gram Konvolüsyonu):** Çok dereceli ($K \in \{1, 3, 5, 9\}$) pencereler ile Türkçe sondan eklemeli morfo-fonetik geçişleri ve koartikülasyonu modeller.
* **Stage 2 (Segmental Zamansal Dikkat Havuzlama, $k=5$):** 50 Hz kareleri 10 Hz'e sıkıştırırken pencere içi ağırlık merkezi havuzlaması (centroid pooling) uygular.
* **Stage 3 (Görmez Çok Kollu Paralel İşleme):**
  - **Kol 1 (Recurrent Acoustic Memory):** Çift yönlü GRU (`GormezBiGRUMemoryBlock`, hidden=256, out=512) ile konuşma dalga boyu momentumunu ve sürekli fonetik akışı modeller.
  - **Kol 2 (Contextual Temporal Attention):** Dual-scale split-heads dikkat (dim=512, 4 heads) ile yerel pencere kısıtları ve global ünlü uyumu/prosodi bağlamını yakalar.
  - **Kol 3 (Residual Identity Shortcut):** Doğrudan projeksiyon + LayerNorm + GELU (dim=512) ile ham akustik özellik kalitesini ve gradyan akışını korur.
* **Stage 4 (Füzyon ve MLP Projeksiyonu):** `[h_rec, h_attn, h_res]` kollarını birleştirir ($1536$ dim) $\to$ LayerNorm $\to$ MLP $\to$ Qwen LLM'in $3584$ boyutlu gömme uzayına eşler.

### 3. Büyük Dil Modeli ve PEFT LoRA İnce Ayarı
* `Qwen/Qwen2.5-7B-Instruct` temel modeli (4-bit NF4 kuantizasyon).
* Dikkat katmanlarına (`q_proj`, `v_proj`, `k_proj`, `o_proj`) $r=16, \alpha=32, \text{dropout}=0.05$ LoRA adaptörleri eklenir.

### 4. Hedefe Yönelik Kuyruk Veri Artırma Boru Hattı (`tail_augmentations.py`)
* **`TailSpeedPerturbation`:** Cümlenin son $\%20 - \%35$'lik kuyruk dilimi $1.15\times - 1.35\times$ hızlandırılır (40 ms linear cross-fade ile dikiş izi / klik sesi önlenir).
* **`TailEnergyDecay`:** Nefes tükenmesini ve fısıltılı konuşmayı simüle etmek için kuyruk genliğine $-3\text{ dB}$ ila $-8\text{ dB}$ sönümleme uygulanır.
* **`TailSpecMasking`:** Yalnızca aktif konuşma kuyruğunun Mel spektrogram karelerinde zaman ve frekans maskelemesi yoğunlaştırılır.

---

## 💾 Kontrol Noktaları ve Ağırlık Yönetimi (Checkpoints)

SLAM-ASR eğitiminde 15 GB'lık ana LLM ağırlıkları asla diske tekrar yazılmaz:
* `projector.pt`: Eğitilen Görmez hibrit projektör ağırlıkları (~150 MB)
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
