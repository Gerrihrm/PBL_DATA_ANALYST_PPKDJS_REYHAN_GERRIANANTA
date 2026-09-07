import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set konfigurasi halaman agar terlihat profesional
st.set_page_config(
    page_title="Employability Career Recommender", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Utama Dashboard
st.title("🎓 Machine Learning Based Student Employability & Career Recommender")
st.markdown("""
Aplikasi ini memprediksi tingkat **kelayakan kerja (Employability)** Anda secara paralel di **5 Rumpun Karir IT Utama** 
menggunakan arsitektur *Local Classifier per Parent* (Multi-Model Random Forest) yang dilengkapi *Local Feature Selection*.
""")
st.write("---")

# 1. LOAD ASSETS (Model, Selector, & Scaler)
@st.cache_resource
def load_ml_assets():
    # Load Scaler Utama
    scaler = joblib.load("scaler.pkl")
    
    # 5 Rumpun Karir Final (Sesuai arahan Instruktur, Business Tech sudah ditiadakan)
    rumpun_list = ['IT Ops', 'Software Dev', 'Infra & DevOps', 'Data Science & AI', 'Cybersecurity']
    
    # Threshold optimal hasil tuning F1-Score kemarin
    thresholds = {
        'IT Ops': 0.40,
        'Software Dev': 0.39,
        'Infra & DevOps': 0.49,
        'Data Science & AI': 0.40,           'Cybersecurity': 0.42
    }
    
    # Load sub-model dan selector masing-masing rumpun
    models = {}
    selectors = {}
    for group in rumpun_list:
        slug = group.lower().replace(' & ', '_').replace(' ', '_')
        models[group] = joblib.load(f"model_rf_f1_{slug}.pkl")
        selectors[group] = joblib.load(f"selector_f1_{slug}.pkl")
        
    return scaler, thresholds, models, selectors, rumpun_list

# Memuat semua aset ML
try:
    scaler, thresholds, models, selectors, rumpun_list = load_ml_assets()
    st.success("✅ Semua sub-model spesialis dan scaler berhasil dimuat!")
except Exception as e:
    st.error(f"⚠️ Gagal memuat aset model (.pkl). Pastikan semua file model, selector, dan scaler diletakkan di folder yang sama. Error: {e}")

# Mapping Ordinal & Kolom Referensi Training
ordinal_mappings = {
    'University_Year': {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4},
    'Academic_Performance': {'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4},
    'English_Proficiency': {'Basic': 1, 'Intermediate': 2, 'Advanced': 3}
}

numeric_cols = [
    'Age', 'Programming_Skill', 'Communication_Skills', 'Teamwork', 
    'Problem_Solving', 'CGPA', 'Projects_Completed', 'Certifications', 
    'Internships', 'Hackathons'
]

# Daftar semua kolom yang diharapkan oleh Scaler saat training (total 25 kolom setelah One-Hot Encoding)
base_cols_reference = [
    'Age', 'University_Year', 'Programming_Skill', 'Communication_Skills', 'Teamwork',
    'Problem_Solving', 'English_Proficiency', 'CGPA', 'Academic_Performance',
    'Projects_Completed', 'Certifications', 'Internships', 'Hackathons',
    'Gender_Male', 'Gender_Other', 
    'Major_Business Analytics', 'Major_Computer Science', 
    'Major_Cybersecurity', 'Major_Data Science', 'Major_Electrical Engineering', 
    'Major_Information Technology', 'Major_Software Engineering',
    'GitHub_Profile_Yes', 
    'LinkedIn_Profile_Yes',
    'Leadership_Experience_Yes'
]
# 2. INPUT FORM (Sidebar - Hanya Menampilkan Fitur Penting & Relevan)
st.sidebar.header("📊 Input Profil Kompetensi")

major = st.sidebar.selectbox(
    "Jurusan", 
    ["Computer Science", "Information Technology", "Software Engineering", 
     "Business Analytics", "Electrical Engineering", "Artificial Intelligence", 
     "Data Science", "Cybersecurity"]
)
english_proficiency = st.sidebar.selectbox(
    "English Proficiency", 
    ["Basic", "Intermediate", "Advanced"],
    index=2  # Default ke 'Advanced'
)
cgpa = st.sidebar.slider("IPK", 0.0, 4.0, 3.2, step=0.1)
prog_skill = st.sidebar.slider("Skor Pemrograman (1-10)", 1, 10, 5)
comm_skill = st.sidebar.slider("Skor Komunikasi (1-10)", 1, 10, 5)
teamwork = st.sidebar.slider("Skor Kerja Sama (1-10)", 1, 10, 5)
prob_solving = st.sidebar.slider("Skor Problem Solving (1-10)", 1, 10, 5)

st.sidebar.subheader("💼 Portofolio & Pengalaman")
projects = st.sidebar.number_input("Jumlah Proyek Selesai", 0, 10, 2)
certs = st.sidebar.number_input("Jumlah Sertifikasi", 0, 10, 1)
internships = st.sidebar.number_input("Jumlah Magang", 0, 5, 0)
hackathons = st.sidebar.number_input("Jumlah Hackathon Diikuti", 0, 10, 0)
age = st.sidebar.number_input("Usia (Age)", 18, 30, 21)


# 3. TOMBOL PROSES & PREPROCESSING DATA INPUT
if st.button("🚀 Uji Kelayakan Kerja & Berikan Rekomendasi"):
    
    # Menyusun data dari form input UI
    input_data = {
        'Age': age,
        'Major': major,
        'Programming_Skill': prog_skill,
        'Communication_Skills': comm_skill,
        'Teamwork': teamwork,
        'Problem_Solving': prob_solving,
        'CGPA': cgpa,
        'Projects_Completed': projects,
        'Certifications': certs,
        'Internships': internships,
        'Hackathons': hackathons,
        'English_Proficiency': english_proficiency 
    }
    
    # TAKTIK INVISIBLE CONSTANTS: Mengisi fitur exclude dengan nilai default (Senior, Good, Male, dsb.)
    input_data['Academic_Performance'] = 'Good'
    input_data['Gender'] = 'Male'
    input_data['University_Year'] = 'Senior'
    input_data['GitHub_Profile'] = 'No'
    input_data['LinkedIn_Profile'] = 'Yes'
    input_data['Leadership_Experience'] = 'Yes'
    
    # Ubah menjadi DataFrame
    df_input = pd.DataFrame([input_data])
    
    # A. Mapping Variabel Ordinal
    for col, mapping in ordinal_mappings.items():
        if col in df_input.columns:
            df_input[col] = df_input[col].map(mapping)
            
    # B. One-Hot Encoding Variabel Nominal
    nominal_cols = ['Gender', 'Major', 'GitHub_Profile', 'LinkedIn_Profile', 'Leadership_Experience']
    df_input_encoded = pd.get_dummies(df_input, columns=nominal_cols)
    
    # C. Penyelarasan Kolom agar Pas dengan base_cols_reference (25 kolom)
    for col in base_cols_reference:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0
    df_input_encoded = df_input_encoded[base_cols_reference]
    
    # D. Standardisasi Fitur Numerik menggunakan Scaler
    df_input_encoded[numeric_cols] = scaler.transform(df_input_encoded[numeric_cols])
    
    # 4. PREDIKSI PARALEL & METRIC CARD LAYOUT
    st.subheader("📋 Dashboard Hasil Kelayakan Kerja")
    
    # Membuat grid 5 kolom (karena sub-model sekarang berjumlah 5)
    cols = st.columns(5)
    
    rekomendasi_utama_ditemukan = False
    
    for idx, group in enumerate(rumpun_list):
        col = cols[idx]
        
        # Ambil Model & Selector khusus rumpun ini
        model_rf = models[group]
        selector = selectors[group]
        threshold_optimal = thresholds[group]
        
        # Saring fitur secara lokal (Local Feature Selection)
        X_selected = selector.transform(df_input_encoded)
        
        # Prediksi Probabilitas Sukses (Placed)
        prob_success = model_rf.predict_proba(X_selected)[0, 1]
        
        # Keputusan Kelulusan berdasarkan threshold optimal
        is_lolos = prob_success >= threshold_optimal
        
        with col:
            # Tampilkan metrik presentase kecocokan
            st.metric(
                label=f"💼 {group}", 
                value=f"{prob_success*100:.1f}%",
                delta=f"Threshold: {threshold_optimal*100:.0f}%",
                delta_color="off"
            )
            
            # Label Keputusan
            if is_lolos:
                st.success("RECOMMENDED")
                rekomendasi_utama_ditemukan = True
            else:
                st.error("NOT ELIGIBLE")
                
    st.write("---")
    
    # 5. KESIMPULAN REKOMENDASI SISTEM (Jujur dan Realistis)
    st.subheader("💡 Analisis & Tindak Lanjut Karir")
    if rekomendasi_utama_ditemukan:
        st.info("""
        🎉 **Rekomendasi Berhasil Dibuat!**  
        Profil Anda dinyatakan memenuhi kualifikasi kelolosan minimum di rumpun bertanda **RECOMMENDED** di atas. 
        Fokuslah untuk mengasah sertifikasi profesional yang relevan guna melipatgandakan peluang daya saing Anda di pasar kerja nyata.
        """)
    else:
        st.warning("""
        🛑 **Sistem Mendeteksi Kesenjangan Kompetensi (Skill Gap)**  
        Profil Anda saat ini **belum mencapai standar kualifikasi minimum** di rumpun karir manapun.  
        
        **Langkah Intervensi yang Disarankan:**
        1. Tingkatkan nilai **IPK (CGPA)** Anda hingga di atas batas minimal kelayakan industri (disarankan di atas 3.00).
        2. Selesaikan minimal **1-2 proyek portofolio mandiri** yang relevan dengan bidang yang Anda minati.
        3. Ikuti program sertifikasi kompetensi industri untuk memperkuat nilai tawar profil Anda.
        """)