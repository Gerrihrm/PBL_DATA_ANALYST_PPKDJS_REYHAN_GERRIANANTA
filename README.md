# 🎓 Sistem Rekomendasi Karir Berbasis *Employability* Menggunakan Pendekatan Multi-Model Random Forest dan *Local Feature Selection*

Sistem rekomendasi interaktif berbasis kecerdasan buatan (*AI*) untuk mengukur tingkat **kelayakan kerja (*Employability*)** mahasiswa di bidang teknologi informasi (IT). Menggunakan pendekatan arsitektur **Local Classifier per Parent (LCP)**, sistem ini tidak sekadar memberikan rekomendasi berdasarkan minat, melainkan melakukan uji kualifikasi kelayakan riil secara paralel di 5 rumpun karir industri utama.

🔗 **Link Web App (Streamlit Cloud):** *https://pbldataanalystppkdjsreyhangerriananta-u2iu6i4rxt7gydztodyajy.streamlit.app/*

---

## 📌 Latar Belakang & Permasalahan

Dalam sistem rekomendasi karir tradisional (*flat/single model classification*), seluruh data dipaksa untuk digeneralisasi menggunakan fitur global. Hal ini menimbulkan dua masalah utama:

1. **Ekspektasi Palsu:** Siswa dengan kualifikasi sangat rendah (misalnya IPK 1.5 dan tanpa keahlian) akan tetap mendapatkan rekomendasi bidang tertentu karena model dipaksa memilih satu pemenang (*multiclass*).
2. **Pengenceran Fitur (*Feature Dilution*):** Standar industri IT sangat heterogen (misalnya rumpun *Software Dev* mengutamakan kemampuan pengodean, sedangkan *Data Science* mengutamakan statistik). Menguji seluruh rumpun dengan fitur global membuat akurasi model menurun secara drastis.

### Solusi Kami:
Proyek PBL ini menerapkan arsitektur **Multi-Model / Local Classifier (LCP)**. Sistem menduplikasi data profil siswa dan mengujinya secara independen di **5 sub-model spesialis** secara paralel. Jika kompetensi siswa memang belum siap kerja, model akan bersikap jujur dengan memberikan keputusan `NOT ELIGIBLE (Tidak Layak)` di bidang terkait.

---

## 🛠️ Arsitektur & Rekayasa Data

Proyek ini dikembangkan secara metodologis dengan standar industri *Machine Learning Pipeline*:

1. **Dataset:** Menggunakan 50.000 data rekam akademis dan portofolio profesional siswa.
2. **SMOTE Lokal (Localized SMOTE):** Menangani masalah ketidakseimbangan kelas (*class imbalance*) secara terisolasi pada masing-masing sub-model guna menghindari kebocoran data.
3. **Seleksi Fitur Lokal (*Local Feature Selection*):** Menggunakan `SelectFromModel` berbasis *Gini Importance* (threshold "mean") untuk menyaring dan membuang fitur non-relevan di tiap pintu masuk sub-model.
4. **Kalibrasi F1-Threshold:** Threshold keputusan kelulusan (*Placed*/*Not Placed*) dioptimalkan menggunakan probabilitas *Out-of-Bag (OOB)* untuk mendapatkan keseimbangan *Precision* & *Recall* terbaik tanpa bias.

---

## 📊 Hasil Evaluasi Model Final (Tuned)

Berikut adalah performa model final spesialis kami setelah proses tuning:

| Sub-Model Rumpun Karir | Jumlah Fitur | Threshold Optimal | Akurasi Uji | F1-Score Uji | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IT Ops** | 12 Fitur | 0.40 | 75.16% | 81.82% | 86.50% |
| **Software Dev** | 10 Fitur | 0.39 | 80.25% | 88.46% | 75.43% |
| **Infra & DevOps** | 7 Fitur | 0.49 | 90.56% | 94.28% | 96.33% |
| **Data Science & AI** | 11 Fitur | 0.40 | 85.74% | 92.06% | 76.80% |
| **Cybersecurity** | 10 Fitur | 0.42 | 80.18% | 88.28% | 74.82% |

*Catatan: Sesuai arahan instruktur industri, bidang **Business Analyst** telah diintegrasikan sepenuhnya ke dalam sub-model **Data Science & AI**.*

---

## 📂 Struktur Repositori

```text
├── app.py                         # Kode utama antarmuka Streamlit Web App
├── global_scaler.pkl              # Standarisasi fitur numerik dari Colab
├── requirements.txt               # Daftar dependensi library Python
├── model_rf_final_*.pkl           # 5 File model biner Random Forest (.pkl)
├── selector_final_*.pkl           # 5 File selector fitur lokal (.pkl)
├── data/
│   └── dataset_career.csv         # Raw dataset 50.000 records
└── notebooks/
    └── PBL_Career_Prediction.ipynb # Jupyter Notebook dokumentasi eksperimen Colab
```
⚙️ Cara Menjalankan Aplikasi Secara Lokal
``` text
1. Clone Repositori ini:
    git clone https://github.com/Gerrihrm/PBL_DATA_ANALYST_PPKDJS_REYHAN_GERRIANANTA.git
    cd PBL_DATA_ANALYST_PPKDJS_REYHAN_GERRIANANTA
2. Instal Library yang Dibutuhkan:
    pip install -r requirements.txt
3. Jalankan Aplikasi Streamlit:
    python -m streamlit run app.py
```
💻 Fitur Antarmuka Pengguna (UI/UX)
**Input Ringkas (11 Fitur Utama): Form input dirancang efisien dengan menyaring fitur yang tidak berkontribusi signifikan pada model, mengoptimalkan waktu pengisian formulir.
**Taktik Invisible Constants: Menyisipkan 6 fitur sekunder secara konstan di balik layar guna mempertahankan bentuk dimensi data input tanpa membebani antarmuka pengguna.
**Analisis & Tindak Lanjut Karir: Sistem memberikan saran rekomendasi peningkatan kompetensi (skill gap intervention) jika nilai kelayakan kerja pengguna di bawah standar minimum industri.
**Dibuat untuk memenuhi tugas Proyek Pembelajaran Berbasis Masalah (PBL) - Program Data Analyst PPKDJS.
---
