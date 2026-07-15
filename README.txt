# Final Project UAS - Prediksi Harga Mobil Bekas

## 📌 Deskripsi Proyek
Proyek ini adalah implementasi **Mini Machine Learning** untuk memprediksi harga mobil bekas di pasar AS menggunakan algoritma **Regresi**. Dataset yang digunakan adalah **US Cars Dataset** dari Kaggle.

## 🎯 Tujuan
- Memprediksi harga jual mobil bekas berdasarkan fitur-fitur yang tersedia
- Membandingkan performa 3 model regresi: **Linear Regression**, **Random Forest (Default)**, dan **Random Forest (Tuned)**
- Mengetahui fitur-fitur yang paling berpengaruh terhadap harga mobil

## 📊 Dataset
- **Sumber**: [US Cars Dataset - Kaggle](https://www.kaggle.com/datasets/doaaalsenani/usa-cers-dataset)
- **Jumlah Data Awal**: 2.499 baris
- **Jumlah Data Setelah Cleaning**: 2.359 baris (140 outlier dihapus)
- **Fitur**: 13 kolom (price, brand, model, year, mileage, title_status, color, state, vin, lot, country, condition, Unnamed: 0)

## 🛠️ Metodologi
1. **Handling Outlier**: Menghapus data dengan harga di bawah $500 dan di atas $48.589 (metode IQR)
2. **Feature Engineering**: Membuat fitur baru `car_age` dan `mileage_per_year`
3. **Preprocessing**: One-Hot Encoding untuk fitur kategorikal, StandardScaler untuk fitur numerik
4. **Modeling**: 
   - Linear Regression (Baseline)
   - Random Forest (Default)
   - Random Forest (Tuned dengan GridSearchCV)
5. **Evaluasi**: R² Score, MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)

## 📈 Hasil Performa Model
| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| **Linear Regression** | **0.6793** 🏆 | $4,088.44 | $5,904.32 |
| Random Forest (Default) | 0.6540 | $4,097.71 | $6,133.36 |
| Random Forest (Tuned) | 0.6472 | $4,186.04 | $6,192.59 |

## 🔍 Insight Utama
- **Fitur terpenting**: `mileage` (27.2%), `mileage_per_year` (7.1%), `brand_Nissan` (6.5%)
- **Korelasi**: Harga berkorelasi positif dengan `year` (0.38) dan negatif dengan `mileage` (-0.39)
- **Model terbaik**: Linear Regression dengan R² = 0.6793

## 📂 Struktur Folder

FinalProject/
├── data/
│   └── USA_cars_datasets.csv
├── output/
│   ├── feature_importance_final.png
│   ├── heatmap_final.png
│   ├── rf_tuned.pkl
│   ├── scaler.pkl
│   └── scatter_comparison.png
├── myenv/               (virtual environment)
├── main.py              ✅ FILE UTAMA (EKSEKUSI)
├── utils.py             ✅ FILE MODUL (FUNGSI) ← BUAT INI DULU!
├── requirements.txt
├── README.txt           (ganti dari .md)
└── laporan_UAS.pdf

## 🚀 Cara Menjalankan
1. **Aktifkan virtual environment**:
   ```bash
   myenv\Scripts\activate.bat
2. Install dependencies (jika belum):
    ```bash
    pip install -r requirements.txt
3. Jalankan program:
    ```bash
    python main.py

### 📹 Video Dokumentasi
[Link YouTube akan diisi]

### 🔗 Link GitHub
[Link GitHub akan diisi]

### 👤 Identitas
    Nama: [FERDY ANDHIKA TANGKEALLO]
    NIM: [25.11.6364]
    Kelas: [25IF02]

### Dibuat untuk memenuhi Ujian Akhir Semester - Mini Machine Learning

## 2️⃣ **requirements.txt** (UPDATE)
  
    pandas==2.0.3
    numpy==1.24.3
    matplotlib==3.7.1
    seaborn==0.12.2
    scikit-learn==1.3.0
    joblib==1.3.1
    openpyxl==3.1.2

**Catatan:** Hapus `xgboost` karena di kode kamu tidak menggunakannya (hanya di-import untuk jaga-jaga).

---

## 📋 **CEKLIST YANG HARUS KAMU ISI SENDIRI**

| Item | Di mana? | Yang harus diisi |
| :--- | :--- | :--- |
| Nama | README.md & Laporan | Nama lengkap kamu |
| NIM | README.md & Laporan | NIM kamu |
| Kelas | README.md & Laporan | Kelas kamu |
| Link GitHub | README.md & Laporan | Link repo GitHub setelah upload |
| Link YouTube | README.md & Laporan | Link video setelah upload |
| Screenshot kode | Laporan | Screenshot dari VSCode |
| Screenshot output | Laporan | Screenshot dari terminal |

---

## 🚀 **LANGKAH SELANJUTNYA**

1. **Update README.md** → Copy-paste isi di atas, isi nama/NIM/link
2. **Update requirements.txt** → Copy-paste isi di atas
3. **Buat laporan** → Copy-paste kerangka ke Word/Google Docs
4. **Tambahkan screenshot** → Ambil screenshot dari VSCode dan terminal
5. **Export ke PDF** → Save as PDF
6. **Upload ke GitHub** → Saya bantu ajarin kalau belum tahu
7. **Buat video YouTube** → Saya bantu kerangka script kalau mau
