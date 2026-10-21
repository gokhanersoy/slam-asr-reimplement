# Turkish SLAM-ASR: Hedefe Yönelik Kuyruk Veri Artırma Ablasyonu (t1-tail-aug)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin **Türkçe** konuşma tanıma uyarlamasıdır.

Bu proje (`t1-tail-aug`), Türkçe konuşma tanımada sıkça gözlenen **cümle sonundaki kelimelerin ve çekim eklerinin yutulması (phoneme swallowing / word tail truncation)** ve nefes tükenmesine bağlı enerji düşüşü problemine karşı geliştirilmiş **Hedefe Yönelik Kuyruk Veri Artırma (Targeted Tail Data Augmentation)** ablasyon çalışmasını içerir.

---

## 🎯 Akustik Problem ve Çözüm Hipotezi

* **Fonetik Gerçeklik:** Sondan eklemeli dillerde (Türkçe) konuşmacılar cümlenin yüklemini ve eklerini söylerken akciğerdeki hava basıncı düşer. Cümleyi bitirmek için **son 1-2 saniyede istemsizce hızlanılır**, fonemler iç içe geçer ve ses enerjisi zayıflar.
* **Akustik Kare Sıkışması:** Whisper ses kodlayıcısı 50 Hz ile çalışırken projektör bunu 5 kat ($k=5$) sıkıştırır. Normalde 400 ms süren bir ek, hızlı söylendiğinde 150 ms'ye düşer ve geriye sadece 1-2 gömme vektörü kalır. Model antrenmanda bunu görmezse ekleri ve son kelimeleri **tahmin edemeyip kesip atar.**
* **Çözüm Hipotezi:** Eğitim sırasında seslerin sadece son $\%20 - \%35$'lik kuyruk bölümünü yapay olarak $\%15 - \%35$ hızlandırıp hafifçe sönümlendirerek modelin sıkışmış akustik karelerden ve zayıf enerjiden doğru transkript çıkarması sağlanır.

---

## 📊 Deneysel Karşılaştırma Tablosu

| Deney Kodu | Projektör Mimarisi | LLM Durumu | Veri Artırma / Ön İşleme | ISSAI TSC (WER) | Mozilla CV-TR (WER) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`t0` (Temel / Baseline)** | Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Yok (Standart) | %12.98 | %14.85 |
| **`t1-preprocess`** | Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Akustik Kuyruk Koruyucu + DRC | %13.29 | %15.01 |
| **`t1-feature-aug`** | Doğrusal ($k=5$) | Dondurulmuş (Frozen) | Genel Hız Değişimi + SpecAugment | %13.17 | %15.22 |
| **`t1-tail-aug` (Bu Proje)** | **Doğrusal ($k=5$)** | **Dondurulmuş (Frozen)** | **Hedefe Yönelik Kuyruk Hızlandırma + Sönümleme** | *Eğitilecek* | *Eğitilecek* |
| **`t4-ng-ta-gormez-lora`** | Görmez Hibrit | PEFT LoRA ($r=16, \alpha=32$) | Yok (Standart) | **%11.53** | **%13.16** |

---

## 🏗️ Kuyruk Veri Artırma Boru Hattı (`tail_augmentations.py`)

1. **`TailSpeedPerturbation` (Kuyruk Hızlandırma):**
   - Cümlenin son $\%20 - \%35$'lik dilimi $1.15\times - 1.35\times$ hızlandırılır.
   - Gövde ile hızlanan kuyruk arasına $40\text{ ms}$ lineer cross-fade uygulanarak faz sıçramaları ve klik sesleri önlenir.
2. **`TailEnergyDecay` (Kuyruk Enerji Sönümlemesi):**
   - Son bölüme $-3\text{ dB}$ ila $-8\text{ dB}$ doğrusal genlik sönümlemesi (attenuation ramp) uygulanarak nefes bitimi ve fısıltılı konuşma simüle edilir.
3. **`TailSpecMasking` (Kuyruk Spektrogram Maskelemesi):**
   - Yalnızca aktif konuşma kuyruğunun log-Mel karelerinde zaman/frekans maskelemesi yoğunlaştırılarak modelin eksik fonetik ipuçlarına dayanıklılığı artırılır.

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
