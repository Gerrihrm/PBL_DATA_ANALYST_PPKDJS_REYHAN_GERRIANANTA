🎓 Rancang Bangun Sistem Rekomendasi Karir Berbasis Employability Menggunakan Pendekatan Multi-Model Random Forest dan Local Feature Selection
Sistem rekomendasi interaktif berbasis kecerdasan buatan (AI) untuk mengukur tingkat kelayakan kerja (Employability) mahasiswa di bidang teknologi informasi (IT) [14]. Menggunakan pendekatan arsitektur Local Classifier per Parent (LCP), sistem ini tidak sekadar memberikan rekomendasi berdasarkan minat, melainkan melakukan uji kualifikasi kelayakan riil secara paralel di 5 rumpun karir industri utama [12, 14, 17].
🔗 Link Web App (Streamlit Cloud): [Tempel link Streamlit Anda di sini, misal: https://career-recommender.streamlit.app/]
📌 Latar Belakang & Permasalahan
Dalam sistem rekomendasi karir tradisional (flat/single model classification), seluruh data dipaksa untuk digeneralisasi menggunakan fitur global [12]. Hal ini menimbulkan dua masalah utama:
Ekspektasi Palsu: Siswa dengan kualifikasi sangat rendah (misalnya IPK 1.5 dan tanpa keahlian) akan tetap mendapatkan rekomendasi bidang tertentu karena model dipaksa memilih satu pemenang (multiclass).
Pengenceran Fitur (Feature Dilution): Standar industri IT sangat heterogen (misalnya rumpun Software Dev mengutamakan kemampuan pengodean, sedangkan Data Science mengutamakan statistik) [12]. Menguji seluruh rumpun dengan fitur global membuat akurasi model menurun secara drastis [7].
Solusi Kami:
Proyek PBL ini menerapkan arsitektur Multi-Model / Local Classifier (LCP) [12, 17]. Sistem menduplikasi data profil siswa dan mengujinya secara independen di 5 sub-model spesialis secara paralel [14]. Jika kompetensi siswa memang belum siap kerja, model akan bersikap jujur dengan memberikan keputusan NOT ELIGIBLE (Tidak Layak) di bidang terkait [14].
🛠️ Arsitektur & Rekayasa Data
Proyek ini dikembangkan secara metodologis dengan standar industri Machine Learning Pipeline [12]:
Dataset: Menggunakan 50.000 data rekam akademis dan portofolio profesional siswa [10].
SMOTE Lokal (Localized SMOTE): Menangani masalah ketidakseimbangan kelas (class imbalance) secara terisolasi pada masing-masing sub-model guna menghindari kebocoran data [12].
Seleksi Fitur Lokal (Local Feature Selection): Menggunakan SelectFromModel berbasis Gini Importance (threshold "mean") untuk menyaring dan membuang fitur non-relevan di tiap pintu masuk sub-model [12].
Kalibrasi F1-Threshold: Threshold keputusan kelulusan (Placed/Not Placed) dioptimalkan menggunakan probabilitas Out-of-Bag (OOB) untuk mendapatkan keseimbangan Precision & Recall terbaik tanpa bias [12].
📊 Hasil Evaluasi Model Final (Tuned)
Melalui kombinasi tuning parameter menggunakan RandomizedSearchCV dan optimasi threshold kustom, berikut adalah performa model final spesialis kami [12]:
Sub-Model Rumpun Karir
Jumlah Fitur
Threshold Optimal
Akurasi Uji
F1-Score Uji
AUC-ROC
IT Ops
12 Fitur
0.40
75.16%
81.82%
86.50%
Software Dev
10 Fitur
0.39
80.25%
88.46%
75.43%
Infra & DevOps
7 Fitur
0.49
90.56%
94.28%
96.33%
Data Science & AI
11 Fitur
0.40
85.74%
92.06%
76.80%
Cybersecurity
10 Fitur
0.42
80.18%
88.28%
74.82%
Catatan: Sesuai arahan instruktur industri, bidang Business Analyst telah diintegrasikan sepenuhnya ke dalam sub-model Data Science & AI.
📂 Struktur Repositori
├── app.py                         # Kode utama antarmuka Streamlit Web App
├── global_scaler.pkl              # Standarisasi fitur numerik dari Colab
├── requirements.txt               # Daftar dependensi library Python
├── model_rf_final_*.pkl           # 5 File model biner Random Forest (.pkl)
├── selector_final_*.pkl           # 5 File selector fitur lokal (.pkl)
├── data/
│   └── dataset_career.csv         # Raw dataset 50.000 records
└── notebooks/
    └── PBL_Career_Prediction.ipynb # Jupyter Notebook dokumentasi eksperimen Colab
⚙️ Cara Menjalankan Aplikasi Secara Lokal
Clone Repositori ini:
git clone https://github.com/Gerrihrm/PBL_DATA_ANALYST_PPKDJS_REYHAN_GERRIANANTA.git
cd PBL_DATA_ANALYST_PPKDJS_REYHAN_GERRIANANTA
Instal Library yang Dibutuhkan:
pip install -r requirements.txt
Jalankan Aplikasi Streamlit:
python -m streamlit run app.py
💻 Fitur Antarmuka Pengguna (UI/UX)
Input Ringkas (11 Fitur Utama): Form input dirancang efisien dengan menyaring fitur yang tidak berkontribusi signifikan pada model, mengoptimalkan waktu pengisian formulir.
Taktik Invisible Constants: Menyisipkan 6 fitur sekunder secara konstan di balik layar guna mempertahankan bentuk dimensi data input tanpa membebani antarmuka pengguna [14].
Analisis & Tindak Lanjut Karir: Sistem memberikan saran rekomendasi peningkatan kompetensi (skill gap intervention) jika nilai kelayakan kerja pengguna di bawah standar minimum industri [14].
Dibuat untuk memenuhi tugas Proyek Pembelajaran Berbasis Masalah (PBL) - Program Data Analyst PPKDJS.
