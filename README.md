# Proyek Analisis Sentimen Teks Bahasa Indonesia

Repositori ini berisi implementasi pipeline Machine Learning berbasis Natural Language Processing (NLP) untuk menganalisis dan mengklasifikasikan sentimen teks ulasan (positif atau negatif) dalam Bahasa Indonesia.

---

## 📁 Struktur Proyek

```text
sentiment-analysis/
│
├── data/
│   └── ulasan.csv                   # Dataset ulasan sentimen Bahasa Indonesia
│
├── model/
│   └── model_sentimen.joblib        # Model terlatih (Pipeline TF-IDF + Logistic Regression)
│
├── notebook.ipynb                   # Jupyter Notebook utama (alur lengkap ML & visualisasi)
├── sentiment_analysis.ipynb         # Salinan deskriptif untuk notebook analisis sentimen
├── requirements.txt                 # Daftar dependensi & pustaka yang diperlukan
├── .gitignore                       # File pengabaian git untuk Python & Jupyter
└── README.md                        # Dokumentasi proyek
```

---

## 🛠️ Pustaka & Dependensi

Proyek ini menggunakan pustaka-pustaka Python berikut:

- **Pandas** (`pandas`): Manipulasi dan analisis data tabular.
- **Scikit-Learn** (`scikit-learn`): Pra-pemrosesan teks (TF-IDF Vectorizer), algoritma klasifikasi (Logistic Regression), dan metrik evaluasi.
- **NumPy** (`numpy`): Operasi komputasi numerik dan array.
- **Joblib** (`joblib`): Serialisasi dan deserialisasi model Machine Learning untuk inferensi cepat.
- **Datasets** (`datasets`): Akses dan pemuatan dataset NLP skala besar.
- **Requests & Gzip** (`requests`, `gzip`): Pengambilan data via HTTP dan pembacaan berkas arsip berukuran besar secara streaming.
- **Matplotlib & Seaborn** (`matplotlib`, `seaborn`): Visualisasi data dan confusion matrix.

---

## 🚀 Panduan Instalasi & Persiapan Lingkungan

### 1. Klon Repositori
```bash
git clone https://github.com/Raihan220106/sentiment-analysis.git
cd sentiment-analysis
```

### 2. (Opsional) Buat dan Aktifkan Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

---

## 💡 Cara Menjalankan

### Menjalankan Jupyter Notebook
Buka dan jalankan `notebook.ipynb` atau `sentiment_analysis.ipynb` di Jupyter Lab / Jupyter Notebook / VS Code:
```bash
jupyter notebook notebook.ipynb
```

Setiap sel diatur secara berurutan:
1. **Pemuatan Pustaka**
2. **Pemuatan & Eksplorasi Dataset (`data/ulasan.csv`)**
3. **Exploratory Data Analysis (EDA)**
4. **Pembagian Data Train/Test (80:20)**
5. **Pelatihan Model & Pipeline (TF-IDF + Logistic Regression)**
6. **Evaluasi Model (Akurasi ~93%, Precision, Recall, F1-Score, Confusion Matrix)**
7. **Penyimpanan Model ke `model/model_sentimen.joblib`**
8. **Pengujian Inferensi Real-time**

### Menggunakan Model Terlatih Langsung (Python Script)
Anda dapat memuat dan menguji model tersimpan secara instan dengan kode berikut:

```python
import joblib

# Muat model
model = joblib.load('model/model_sentimen.joblib')

# Kalimat uji
kalimat = [
    "Pelayanan sangat memuaskan dan produk cepat sampai!",
    "Barang rusak, penjual lambat merespon dan sangat mengecewakan."
]

# Prediksi (0 = Negatif, 1 = Positif)
prediksi = model.predict(kalimat)
probabilitas = model.predict_proba(kalimat)

for teks, label, prob in zip(kalimat, prediksi, probabilitas):
    status = "POSITIF" if label == 1 else "NEGATIF"
    print(f"Teks      : {teks}")
    print(f"Sentimen  : {status} ({max(prob)*100:.1f}%)")
    print("-" * 50)
```

---

## 📊 Hasil Evaluasi

Model mencapai akurasi **93%** pada data uji:
- **Precision**: 0.91 (Negatif), 0.96 (Positif)
- **Recall**: 0.96 (Negatif), 0.90 (Positif)
- **F1-Score**: 0.93 (Negatif), 0.93 (Positif)

---

## 👤 Penulis
- **Raihan Muhammad Riswandi** ([@Raihan220106](https://github.com/Raihan220106))
