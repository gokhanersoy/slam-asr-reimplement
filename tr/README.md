# SLAM-ASR: Türkçe Konuşma Tanıma (Turkish Speech Recognition)

PyTorch ve Hugging Face tabanlı, **SLAM-ASR** (*arXiv:2402.08846*) mimarisinin Türkçe konuşma tanıma uyarlamalarıdır.

Bu dizin altında iki bağımsız Türkçe veri seti çalışması yer almaktadır:

---

## 📁 Alt Projeler / Sub-projects

### 1. `tsc/` — ISSAI Turkish Speech Corpus (ISSAI_TSC_218)
* **Veri Seti:** Nazarbayev Üniversitesi ISSAI tarafından hazırlanan 218 saatlik Türkçe konuşma korpusu.
* **Format:** 186.171 ses kaydı (16 kHz WAV) ve metin eşleşmeleri.
* **Elde Edilen Başarı:** Whisper-small + Qwen2.5-7B ile **%23.04 WER** ve **%8.27 CER** (Yalnızca %0.6 parametre eğitilerek).
* **Detaylar:** [tr/tsc/README.md](tsc/README.md)

### 2. `cv/` — Mozilla Common Voice Türkçe
* **Veri Seti:** Mozilla Data Collective (MDC) üzerinden sağlanan Common Voice Türkçe ses korpusu.
* **MDC Dataset ID:** `cmu5wkah500c2o10719lem8m8`
* **Format:** MP3 ses kayıtları (`clips/`) ve TSV etiketleri (`train.tsv`, `dev.tsv`, `test.tsv`).
* **Özellikler:** Colab Secret (`MDC_API_KEY`) entegrasyonu, otomatik akış indirme, Türkçe karakter duyarlı normalizasyon, donanıma duyarlı eğitim ve canlı mikrofon çıkarımı.
* **Detaylar:** [tr/cv/README.md](cv/README.md)

---

## 🏗️ Ortak Mimari Özellikleri

* **Akustik Kodlayıcı (Speech Encoder):** `openai/whisper-small` (Dondurulmuş / Frozen).
* **Büyük Dil Modeli (LLM):** `Qwen/Qwen2.5-7B-Instruct` (Dondurulmuş / Frozen).
* **Projektör (Projector):** $k=5$ zaman boyutunda ardışık kare birleştirme (downsampling, 50 Hz $\to$ 10 Hz) ve tek bir Lineer katman (Eğitilen tek modül).
* **Donanım Uyarlaması:** H100, A100, L4, T4 ve CPU ortamlarına göre otomatik hassasiyet (FP32 / BF16 / 4-bit NF4) ve batch size optimizasyonu.
