# Turkish SLAM-ASR: Görmez Hibrit Projektör + PEFT LoRA + Kuyruk Veri Artırma + Zemberek Dil Modeli (t6-ng-ta-ctx-tail-aug-lora-lm)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** mimarisinin **Türkçe** konuşma tanıma uyarlaması ve **Zemberek Morfolojik Dil Modeli (Language Model - LM)** entegrasyonudur.

Bu proje (`t6-ng-ta-ctx-tail-aug-lora-lm`), tüm tez çalışmalarınız boyunca geliştirilen şampiyon modelimiz (`t5`) üzerine **Zemberek kural tabanlı morfolojik dil modeli ve N-Gram yeniden puanlama (rescoring)** mekanizmasını entegre eden ileri düzey araştırma varyantıdır.

---

## 🔬 Bilimsel Gerekçe ve Danışman Vizyonu

Literatürde Görmez (2024) (*PeerJ CS, e1981*), hibrit bir derin öğrenme akustik modelini (CNN-BiLSTM-GRU-Transformer) kural tabanlı Zemberek morfolojik dil modeli ile destekleyerek kelime hata oranını (WER) %11.50'den %8.40'a indirmeyi başarmıştır. 

Danışmanınızın talebi doğrultusunda geliştirilen bu varyant (**t6**), şu temel bilimsel soruyu deneysel olarak yanıtlamayı hedefler:
> *8 milyar parametreli üretken ve anlamsal muhakeme yeteneğine sahip bir Büyük Dil Modeli (LLM + LoRA) ile oluşturulan uçtan uca çok modlu ASR mimarisinin üzerine harici, sembolik bir Zemberek Dil Modeli (LM) eklendiğinde akustik-morfolojik başarı artar mı, yoksa sözlük kısıtları alan kaymalarında (out-of-domain) hata yayılımına mı yol açar?*

---

## 🎯 Evrimsel Ablasyon ve Karşılaştırma Matrisi (Evolutionary Benchmark Matrix)

| Deney Kodu | Projektör Mimarisi | LLM Durumu | Kuyruk Veri Artırma | Dil Modeli (LM) | ISSAI TSC (WER / CER) | Common Voice (WER / CER) | Bilimsel Rolü |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`t0` (Temel)** | Standart Doğrusal ($k=5$) | Dondurulmuş | Yok | Yok | %13.22 / %3.88 | %14.87 / %4.08 | Orijinal SLAM-ASR temeli |
| **`t1-ng`** | NeMo Kapılı Conv1D | Dondurulmuş | Yok | Yok | %12.49 / %3.68 | %14.33 / %3.76 | Güçlü fonetik ilerleme |
| **`t3-ng-ta-ctx`** | Görmez Hibrit | Dondurulmuş | Yok | Yok | %12.27 / %3.46 | %14.04 / %4.05 | Saf mimari bağlam kazancı |
| **`t1-lora`** | Standart Doğrusal ($k=5$) | PEFT LoRA ($r=16$) | Yok | Yok | %11.78 / %3.40 | -- / -- | Doğrusal modelde LoRA kazancı |
| **`t1-tail-aug-lora`** | Standart Doğrusal ($k=5$) | PEFT LoRA ($r=16$) | Hızlandırma + Sönümleme | Yok | %11.58 / %3.15 | %12.82 / %3.20 | Kuyruk sınır koruma kazancı |
| **`t4-ng-ta-ctx-lora`** | Görmez Hibrit | PEFT LoRA ($r=16$) | Yok | Yok | %11.53 / %3.27 | %13.41 / %3.42 | LLM uyarlama atılımı |
| **`t4-ng-ta-ctx-postcorr`**| Görmez Hibrit | Dondurulmuş | Yok | Arslan Çıktı Düzeltici | %13.60 / %4.02 | %16.99 / %4.99 | Ayrık sözlük son-işleme zaafı |
| **`t5-ng-ta-ctx-tail-aug-lora`** | **Görmez Hibrit** | **PEFT LoRA ($r=16$)** | **Hızlandırma + Sönümleme** | **Yok (Uçtan Uca)** | **%11.43 / %3.21 🏆** | **%12.75 / %3.17 🏆** | **Unified Champion Architecture** |
| **`t6-ng-ta-ctx-tail-aug-lora-lm` (Bu Proje)** | **Görmez Hibrit** | **PEFT LoRA ($r=16$)** | **Hızlandırma + Sönümleme** | **Zemberek LM Rescorer** | **Sınanıyor (t6)** | **Sınanıyor (t6)** | **Zemberek LM Entegrasyonu** |

---

## 🏗️ 5 Temel Gücün Entegrasyonu

1. **Akustik Kodlayıcı (Speech Encoder):** OpenAI Whisper-Large-v3 (1280 dim, [DONDURULMUŞ / FROZEN]). Ham konuşma sinyalinden 50 Hz log-mel filtre bankası öznitelikleri çıkarır.
2. **Görmez Çok Kollu Hibrit Projektör (Projector):**
   * **Stage 1 (Akustik N-Gram Konvolüsyonu):** Çok dereceli ($K \in \{1, 3, 5, 9\}$) pencereler ile Türkçe fonetik geçişleri ve ek sınırlarını yakalar.
   * **Stage 2 (Segmental Zamansal Dikkat Havuzlama, $k=5$):** 50 Hz kareleri 10 Hz'e sıkıştırır ($\mathcal{O}(N^2)$ LLM dikkat maliyetini 25 kat düşürür).
   * **Stage 3 (Görmez Çok Kol):** Çift yönlü GRU (`GormezBiGRU`), Dual-Scale Dikkat (`Split-Heads Attention`) ve Rezidüel Kısayol (`Residual Shortcut`) paralel çalışır.
   * **Stage 4 (Füzyon ve MLP Projeksiyonu):** 1536 dim $\to$ LayerNorm $\to$ MLP $\to$ Qwen LLM'in 3584 boyutlu gömme uzayına eşler.
3. **Büyük Dil Modeli ve PEFT LoRA İnce Ayarı (LLM Backbone):**
   * `Qwen/Qwen2.5-7B-Instruct` temel modeli (4-bit NF4 kuantizasyon).
   * Dikkat projeksiyonlarına (`q_proj`, `v_proj`, `k_proj`, `o_proj`) $r=16, \alpha=32$ LoRA adaptörleri bağlanır.
4. **Hedefe Yönelik Kuyruk Veri Artırma Boru Hattı (`tail_augmentations.py`):**
   * **Kuyruk Hızlandırma (`TailSpeedPerturbation`):** Cümlenin son $\%20 - \%35$'lik kısmı $1.15\times - 1.35\times$ hızlandırılır (40 ms linear cross-fade).
   * **Kuyruk Enerji Sönümlemesi (`TailEnergyDecay`):** Konuşma sonundaki enerji düşüşlerini simüle eden $-3\text{ dB}$ ila $-8\text{ dB}$ sönümleme rampası.
   * **Kuyruk Spektrogram Maskelemesi (`TailSpecMasking`):** Yalnızca aktif konuşma kuyruğunda zaman ve frekans maskelemesi.
5. **Zemberek Morfolojik Dil Modeli Modülü (`slam_asr/utils/zemberek_lm.py`):**
   * **Çift Modlu Çalışma:** Hem yerel `JPype1` + `zemberek-full.jar` Java kütüphanesini hem de sıfır-bağımlılıklı saf Python Türkçe morfolojik n-gram dil modeli motorunu destekler.
   * **Türkçe Ünlü Uyumları & Fonotaktik Denetimi:** Büyük ve küçük ünlü uyumu, ünsüz yumuşaması ve ek hiyerarşisi doğrulaması.
   * **N-Gram Olasılık Puanlaması:** ISSAI_TSC eğitim kümesinden öğrenilen 1-gram, 2-gram ve 3-gram geçiş olasılıkları (Laplace / Jelinek-Mercer yumuşatması).
   * **Akustik-Dilsel Ortak Puanlama:**
     $$\text{Puan} = \text{AkustikFidelity}(w, w') + \lambda_{\text{LM}} \cdot \log P_{\text{Zemberek}}(w' \mid context) + \text{MorphBonus}(w')$$

---

## 🚀 Hızlı Başlangıç

### 1. Testleri Koşturma
```bash
python run_tests.py
```
*(Zemberek LM, fonotaktik kurallar ve n-gram puanlama dahil 56 birim testin tümü otomatik koşturulur).*

### 2. Eğitimi Başlatma
* **Donanıma göre otomatik optimize edilen konfigürasyon ile:**
  ```bash
  python train.py --config config/hardware_adaptive_turkish.yaml
  ```
* **Google Colab (T4 / A100) üzerinde:**
  ```bash
  python train.py --config config/colab_t4_turkish.yaml
  ```

### 3. Değerlendirme (Zemberek LM ile ve LM'siz Karşılaştırmalı WER/CER)
```bash
python evaluate.py \
    --config config/hardware_adaptive_turkish.yaml \
    --checkpoint outputs/slam_asr_turkish_adaptive_t6_lm/best_checkpoint \
    --manifest data/issai_tsc/test.jsonl \
    --language tr \
    --beam_size 4 \
    --use_zemberek_lm \
    --ngram_order 3
```
*Çıktıda hem **RAW ASR (t5)** hem de **WITH ZEMBEREK LM (t6)** metrikleri yan yana raporlanır.*

### 4. Bağımsız Zemberek LM Yeniden Puanlama Betiği
Mevcut herhangi bir `eval_results.jsonl` dosyasını Zemberek ile sonradan puanlamak için:
```bash
python scripts/apply_zemberek_lm.py \
    --input eval_results.jsonl \
    --output eval_results_zemberek_lm.jsonl \
    --train_manifest data/issai_tsc/train.jsonl \
    --ngram_order 3
```
