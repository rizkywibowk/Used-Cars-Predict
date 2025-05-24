import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import re

# --- FUNGSI GLOBAL UNTUK EKSTRAKSI HP DAN LITER ---
def extract_hp(x):
    if pd.isnull(x): return np.nan
    hp = re.findall(r'(\d{2,4})\.?0?HP', str(x))
    return float(hp[0]) if hp else np.nan

def extract_L(x):
    if pd.isnull(x): return np.nan
    l = re.findall(r'(\d\.\d+)L', str(x))
    return float(l[0]) if l else np.nan

# 1. LOAD DATA
@st.cache_data
def load_data():
    # Muat data dan segera lakukan pembersihan dasar untuk kolom yang akan digunakan untuk bounds
    df_loaded = pd.read_csv('used_cars.csv')
    # Minimal cleaning untuk model_year dan milage agar bisa dapat min/max
    df_loaded['milage_numeric'] = pd.to_numeric(
        df_loaded['milage'].astype(str).str.replace('[\,mi. ]', '', regex=True).replace('-', np.nan),
        errors='coerce'
    )
    df_loaded['model_year_numeric'] = pd.to_numeric(df_loaded['model_year'], errors='coerce')
    return df_loaded

df_initial = load_data() # Data awal, sudah ada kolom numeric untuk bounds
df = df_initial.copy() # Bekerja dengan copy untuk preprocessing lebih lanjut

# 2. CLEANING & FEATURE ENGINEERING
def clean_preprocess(df_to_process):
    # Clean price
    df_to_process['price'] = df_to_process['price'].astype(str).str.replace('[\$,]', '', regex=True).str.replace(',', '').astype(float)
    # Gunakan kolom milage_numeric yang sudah ada
    df_to_process['milage'] = df_to_process['milage_numeric']
    # Car age (gunakan model_year_numeric)
    current_year = pd.Timestamp.now().year
    df_to_process['car_age'] = current_year - df_to_process['model_year_numeric']
    # Clean engine_hp & engine_L
    df_to_process['engine_hp'] = df_to_process['engine'].apply(extract_hp)
    df_to_process['engine_L'] = df_to_process['engine'].apply(extract_L)
    # Clean accident_reported & has_clean_title
    df_to_process['accident_reported'] = df_to_process['accident'].apply(lambda x: 1 if 'accident' in str(x).lower() else 0)
    df_to_process['has_clean_title'] = df_to_process['clean_title'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    # Handle missing values
    # Kolom model_year dan milage di df_to_process kini adalah versi numeriknya
    cols_to_fill_mean = ['engine_hp', 'engine_L', 'milage', 'car_age']
    for col in cols_to_fill_mean:
        if col in df_to_process.columns:
            df_to_process[col].fillna(df_to_process[col].mean(), inplace=True)

    cols_to_fill_mode = ['fuel_type', 'ext_col', 'int_col', 'brand', 'transmission']
    for col in cols_to_fill_mode:
        if col in df_to_process.columns:
            df_to_process[col].fillna(df_to_process[col].mode()[0], inplace=True)
    
    df_to_process = df_to_process.dropna(subset=['price'])
    # Drop kolom helper _numeric jika sudah tidak diperlukan
    df_to_process = df_to_process.drop(columns=['milage_numeric', 'model_year_numeric'], errors='ignore')
    return df_to_process

df = clean_preprocess(df)

# 3. SEGMENTASI MESIN
def engine_segment(row):
    engine_text = str(row.get('engine', '')).lower()
    fuel = str(row.get('fuel_type', '')).lower()
    if 'electric' in engine_text or fuel == 'electric':
        return 'Electric'
    elif 'hybrid' in engine_text or 'plug-in' in engine_text or 'hybrid' in fuel:
        return 'Hybrid'
    elif 'v12' in engine_text or '12 cylinder' in engine_text:
        return 'V12'
    elif 'v10' in engine_text or '10 cylinder' in engine_text:
        return 'V10'
    elif 'v8' in engine_text or '8 cylinder' in engine_text:
        return 'V8'
    elif 'v6' in engine_text or '6 cylinder' in engine_text:
        return 'V6'
    elif 'i4' in engine_text or '4 cylinder' in engine_text or 'flat 4' in engine_text:
        return 'I4'
    elif 'i3' in engine_text or '3 cylinder' in engine_text:
        return 'I3'
    else:
        return 'Other'

df['engine_segment'] = df.apply(engine_segment, axis=1)

# 4. ENCODING FITUR KATEGORIKAL
categorical_cols_to_encode = ['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'engine_segment']
existing_categorical_cols = [col for col in categorical_cols_to_encode if col in df.columns]
df_encoded = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True)


# 5. SPLIT DATA
base_features = ['car_age', 'milage', 'engine_hp', 'engine_L', 'accident_reported', 'has_clean_title']
encoded_feature_cols = [col for col in df_encoded.columns if any(cat_col in col for cat_col in existing_categorical_cols)]
final_features = base_features + encoded_feature_cols

X = df_encoded[[feat for feat in final_features if feat in df_encoded.columns]]
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. TRAIN MODEL (sekali saja, lalu simpan)
@st.cache_resource
def train_and_save_model(X_train_data, y_train_data):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_data, y_train_data)
    joblib.dump(model, 'rf_model.pkl')
    return model

if 'model' not in st.session_state:
    model = train_and_save_model(X_train, y_train)
    st.session_state.model = model
else:
    model = st.session_state.model


# --- STYLING & HEADER ---
st.markdown("""
    <style>
    /* Light mode (default) */
    body, .main, .stApp {
        background-color: #f2f6fc; /* Latar belakang utama aplikasi */
        color: #0a3d62; /* Warna teks utama */
    }
    .title {color: #0a3d62; font-size: 2.8em; font-weight: bold; text-align: center; margin-bottom: 5px;}
    .subtitle {color: #3c6382; font-size: 1.2em; text-align: center; margin-bottom: 20px;}
    .stButton>button {
        background-color: #0a3d62;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3c6382;
        color: white;
    }
    .stSuccess {
        background-color: #e6fffa;
        border-left: 5px solid #38b2ac;
        color: #234e52;
        padding: 10px;
        border-radius: 5px;
    }
    .footer {text-align: center; color: gray; margin-top: 40px; font-size: 0.9em;}
    h3 {color: #0a3d62; border-bottom: 2px solid #0a3d62; padding-bottom: 5px;}
    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        body, .main, .stApp {
            background-color: #1e1e1e !important; /* Latar belakang gelap */
            color: #e0e0e0 !important; /* Teks terang */
        }
        .title {color: #61dafb !important;}
        .subtitle {color: #a0a0a0 !important;}
        .stButton>button {
            background-color: #61dafb !important;
            color: #1e1e1e !important;
        }
        .stButton>button:hover {
            background-color: #4fa8c7 !important;
            color: #1e1e1e !important;
        }
        .stTextInput>div>div>input, .stSelectbox>div>div, .stNumberInput>div>div>input {
            background-color: #2a2a2a !important;
            color: #e0e0e0 !important;
            border: 1px solid #555555 !important;
        }
        .stSlider>div>div>div {
            color: #e0e0e0 !important;
        }
        .stMarkdown, .stDataFrame, .stTable, h3 {
            color: #e0e0e0 !important;
        }
        .stSuccess {
            background-color: #1a3633 !important;
            border-left: 5px solid #38b2ac !important;
            color: #a0f0ed !important;
        }
        h3 {color: #61dafb !important; border-bottom: 2px solid #61dafb;}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🚗 Prediksi Harga Mobil Bekas 🚙</p>', unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
    Aplikasi ini memprediksi harga mobil bekas menggunakan Machine Learning. Jelajahi faktor-faktor yang mempengaruhi harga dan dapatkan estimasi untuk mobil impian Anda!
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Sumber Data:**
Dataset yang digunakan dalam aplikasi ini diambil dari hasil scraping manual website [cars.com](https://www.cars.com) dan telah diunggah ke Kaggle oleh Taeef Najib:
[Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data).
Dataset ini berisi lebih dari 400.000 entri mobil bekas dengan berbagai fitur.
""")


# --- Sidebar: Eksplorasi Data ---
st.sidebar.header("🔎 Eksplorasi Data")
if st.sidebar.checkbox("Tampilkan Dataframe (100 baris pertama)"):
    st.dataframe(df_initial.head(100), use_container_width=True)

st.sidebar.subheader("Distribusi Harga Mobil Bekas")
fig_hist, ax_hist = plt.subplots(figsize=(7,4))
# Gunakan df (yang sudah bersih dan price-nya float) untuk plot distribusi harga
if 'price' in df.columns and not df['price'].empty:
    sns.histplot(df['price'].dropna(), bins=50, ax=ax_hist, color="#0a3d62", kde=True)
    ax_hist.set_title('Distribusi Harga', fontsize=14)
    ax_hist.set_xlabel('Harga Mobil (USD)', fontsize=12)
    ax_hist.set_ylabel('Jumlah Mobil', fontsize=12)
    ax_hist.tick_params(axis='x', labelsize=10)
    ax_hist.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    st.sidebar.pyplot(fig_hist)
else:
    st.sidebar.warning("Data harga tidak tersedia untuk ditampilkan.")


# --- Main: Form Prediksi ---
st.markdown("### 📝 Masukkan Spesifikasi Mobil Anda")
with st.form("prediksi_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand_options = sorted(df_initial['brand'].dropna().unique()) # Dari df_initial agar opsi lengkap
        brand = st.selectbox("🚗 Merek Mobil", brand_options, help="Pilih merek mobil.")
        
        model_input_key = "model_input_text"
        model_in = st.text_input("모델 Model Mobil", key=model_input_key, help="Masukkan model mobil, contoh: Camry, Civic, F-150.")
        
        # Gunakan kolom _numeric dari df_initial untuk slider bounds yang sudah pasti angka
        min_year_val = df_initial['model_year_numeric'].dropna().min()
        max_year_val = df_initial['model_year_numeric'].dropna().max()
        min_year_slider = int(min_year_val) if pd.notna(min_year_val) else 1970
        max_year_slider = int(max_year_val) if pd.notna(max_year_val) else pd.Timestamp.now().year
        default_year_slider = 2020
        if not (min_year_slider <= default_year_slider <= max_year_slider):
            default_year_slider = max_year_slider # Atau min_year_slider, sesuaikan
        model_year = st.slider("📅 Tahun Pembuatan", min_year_slider, max_year_slider, default_year_slider, help=f"Pilih tahun pembuatan mobil, antara {min_year_slider} dan {max_year_slider}.")
        
        milage_max_val_calc = df_initial['milage_numeric'].dropna().max()
        milage_max_slider = int(milage_max_val_calc + 50000) if pd.notna(milage_max_val_calc) else 1000000
        milage = st.number_input("🛣️ Jarak Tempuh (mil)", min_value=0, max_value=milage_max_slider, value=50000, step=1000, help="Masukkan total jarak tempuh mobil dalam mil.")

    with col2:
        fuel_type_options = sorted(df_initial['fuel_type'].dropna().unique()) # Dari df_initial
        fuel_type = st.selectbox("⛽ Tipe Bahan Bakar", fuel_type_options, help="Pilih tipe bahan bakar mobil.")
        
        transmission_options = sorted(df_initial['transmission'].dropna().unique()) # Dari df_initial
        transmission = st.selectbox("⚙️ Transmisi", transmission_options, help="Pilih jenis transmisi mobil.")
        
        engine = st.text_input("🛠️ Deskripsi Mesin (Contoh: 300.0HP 3.7L V6 Cylinder Engine)", help="Masukkan deskripsi mesin, contoh: 2.0L I4 Turbo.")
        
        accident_reported = st.selectbox("💥 Riwayat Kecelakaan?", ["Tidak", "Ya"], help="Apakah mobil pernah dilaporkan mengalami kecelakaan?")
        
        has_clean_title = st.selectbox("📜 Status Dokumen (Clean Title)?", ["Ya", "Tidak"], help="Apakah mobil memiliki dokumen yang bersih (clean title)?")

    submit_col_left, submit_col_center, submit_col_right = st.columns([2,3,2])
    with submit_col_center:
        submitted = st.form_submit_button("✨ Prediksi Harga Sekarang!")

if submitted:
    with st.spinner("🔍 Menganalisis dan Memprediksi Harga..."):
        current_year_form = pd.Timestamp.now().year
        car_age_input = current_year_form - model_year
        engine_hp_input = extract_hp(engine)
        engine_L_input = extract_L(engine)
        accident_reported_bin_input = 1 if accident_reported == "Ya" else 0
        has_clean_title_bin_input = 1 if has_clean_title == "Ya" else 0
        
        engine_seg_input_data = {'engine': engine, 'fuel_type': fuel_type}
        engine_seg_input = engine_segment(engine_seg_input_data)

        # --- BUAT DATAFRAME DARI INPUT PENGGUNA UNTUK DITAMPILKAN ---
        user_input_display = {
            'Merek': [brand],
            'Model': [model_in if model_in else "-"], # Tampilkan "-" jika model kosong
            'Tahun': [model_year],
            'Jarak Tempuh (mil)': [f"{milage:,}"], # Format dengan koma
            'Bahan Bakar': [fuel_type],
            'Transmisi': [transmission],
            'Deskripsi Mesin': [engine if engine else "-"], # Tampilkan "-" jika deskripsi mesin kosong
            'Riwayat Kecelakaan': [accident_reported],
            'Clean Title': [has_clean_title],
            'Umur Mobil (Tahun)': [car_age_input],
            'Engine HP (est.)': [f"{engine_hp_input:.1f}" if pd.notna(engine_hp_input) else "N/A"],
            'Engine Liter (est.)': [f"{engine_L_input:.1f}" if pd.notna(engine_L_input) else "N/A"],
            'Segmen Mesin (est.)': [engine_seg_input]
        }
        df_user_input_display = pd.DataFrame(user_input_display)
        
        # Tampilkan detail input pengguna dalam expander
        with st.expander("📋 Detail Input yang Anda Masukkan untuk Prediksi", expanded=True):
            st.dataframe(df_user_input_display, use_container_width=True, hide_index=True)
            # atau st.table(df_user_input_display) jika ingin tampilan tabel statis

        # --- PERSIAPAN INPUT UNTUK MODEL (TETAP SAMA) ---
        input_dict = {
            'car_age': car_age_input,
            'milage': milage, # Gunakan nilai milage numerik asli untuk model
            'engine_hp': engine_hp_input if pd.notna(engine_hp_input) else X_train['engine_hp'].mean(),
            'engine_L': engine_L_input if pd.notna(engine_L_input) else X_train['engine_L'].mean(),
            'accident_reported': accident_reported_bin_input,
            'has_clean_title': has_clean_title_bin_input,
        }
        
        for cat_col_original in existing_categorical_cols:
            user_value_for_category = locals().get(cat_col_original.replace('_segment','_input'), locals().get(cat_col_original))
            if cat_col_original == 'engine_segment': user_value_for_category = engine_seg_input

            for encoded_col_name in X_train.columns:
                if encoded_col_name.startswith(f"{cat_col_original}_"):
                    category_value_in_col_name = encoded_col_name.split(f"{cat_col_original}_", 1)[1]
                    input_dict[encoded_col_name] = 1 if user_value_for_category == category_value_in_col_name else 0
        
        for col_in_model in X_train.columns:
            if col_in_model not in input_dict:
                if col_in_model not in base_features:
                     input_dict[col_in_model] = 0

        input_df_for_model = pd.DataFrame([input_dict])[X_train.columns] # DataFrame untuk model
        
        pred = model.predict(input_df_for_model) # Prediksi menggunakan input_df_for_model
        
        st.markdown(f"<div class='stSuccess'>💰 Estimasi harga mobil bekas Anda: <strong>${pred[0]:,.2f}</strong></div>", unsafe_allow_html=True)

# --- Feature Importance ---
# (Kode Feature Importance tetap sama seperti sebelumnya)
st.markdown("### 🌟 Faktor Penentu Harga Mobil (Feature Importance)")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(10)

fig_imp, ax_imp = plt.subplots(figsize=(10,6)) 
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax_imp, palette="viridis") 
ax_imp.set_title('10 Faktor Paling Berpengaruh Terhadap Harga', fontsize=15)
ax_imp.set_xlabel('Tingkat Pengaruh (Feature Importance Score)', fontsize=12)
ax_imp.set_ylabel('Faktor (Fitur)', fontsize=12)
ax_imp.tick_params(axis='x', labelsize=10)
ax_imp.tick_params(axis='y', labelsize=10)
plt.tight_layout()
st.pyplot(fig_imp)

st.markdown("#### Penjelasan Mendalam Setiap Faktor:")
specific_explanations = {
    "milage": "Jarak tempuh yang lebih tinggi umumnya mengindikasikan penggunaan yang lebih intensif dan potensi keausan, sehingga cenderung menurunkan harga jual kembali mobil.",
    "car_age": "Usia mobil adalah faktor depresiasi utama. Semakin tua mobil, nilainya cenderung semakin menurun karena perkembangan teknologi baru dan kondisi fisik yang mungkin menurun.",
    "engine_hp": "Tenaga mesin (Horsepower) seringkali berbanding lurus dengan performa dan segmen mobil. Mobil dengan HP lebih tinggi biasanya lebih mahal karena ditujukan untuk pasar performa atau kemewahan.",
    "engine_L": "Kapasitas mesin (Liter) juga berkorelasi dengan tenaga dan harga. Mesin dengan volume lebih besar sering ditemukan pada mobil mewah atau SUV besar, yang harganya lebih tinggi.",
    "brand_Rolls-Royce": "Merek Rolls-Royce adalah simbol status dan kemewahan tertinggi. Setiap unit seringkali adalah 'bespoke' atau dibuat khusus sesuai pesanan (made-by-order), menggunakan material terbaik, dan dirakit tangan oleh pengrajin ahli. Ini menciptakan nilai eksklusivitas dan seni yang sangat tinggi, menjadikan harganya jauh di atas rata-rata.",
    "brand_Lamborghini": "Merek Lamborghini identik dengan supercar berperforma ekstrim, desain avant-garde, dan produksi terbatas. Eksklusivitas dan teknologi canggihnya membuat harganya sangat tinggi.",
    "brand_Ferrari": "Ferrari adalah ikon mobil sport mewah dengan sejarah balap yang kaya. Setiap modelnya menawarkan performa tinggi, desain menawan, dan status prestisius, yang berkontribusi pada nilai jualnya.",
    "brand_Porsche": "Porsche terkenal dengan rekayasa presisi Jerman, mobil sport yang ikonik (seperti 911), dan keseimbangan antara performa di trek dan kenyamanan harian. Reputasi merek dan kualitasnya menjaga nilai mobil tetap tinggi.",
    "engine_segment_V12": "Mesin V12 adalah konfigurasi mesin yang sangat bertenaga dan kompleks, biasanya hanya ditemukan pada mobil sport paling eksotis atau sedan ultra-mewah. Kelangkaan dan performanya membuat mobil dengan mesin ini sangat mahal.",
    "has_clean_title": "Status 'Clean Title' menunjukkan bahwa mobil tidak memiliki riwayat masalah legal atau kerusakan parah (seperti salvage title). Ini memberikan rasa aman bagi pembeli dan secara signifikan meningkatkan nilai jual mobil.",
    "accident_reported": "Riwayat kecelakaan yang dilaporkan, terutama jika parah, dapat secara signifikan menurunkan nilai mobil karena potensi kerusakan struktural atau masalah tersembunyi."
}

for feature_name, importance_score in feat_imp.items():
    explanation = f"- **{feature_name.replace('_', ' ').title()}**: Memberikan kontribusi sebesar **{importance_score:.4f}** dalam model prediksi harga. "
    
    clean_feature_name_for_dict = feature_name
    # Coba cocokkan dengan kunci spesifik, misal jika feature_name adalah brand_Rolls-Royce
    # kita ingin mencari "brand_Rolls-Royce" di dictionary
    
    # Atau, jika kita ingin mencocokkan bagian dari nama fitur, misal 'milage' dari 'milage'
    # atau 'Rolls-Royce' dari 'brand_Rolls-Royce'
    simplified_key_parts = feature_name.split('_')
    
    matched_explanation = False
    if feature_name in specific_explanations: # Cocokkan nama fitur lengkap dulu
        explanation += specific_explanations[feature_name]
        matched_explanation = True
    else: # Jika tidak cocok, coba cocokkan bagian dari nama
        for part in reversed(simplified_key_parts): # Coba dari bagian terakhir (misal Rolls-Royce dari brand_Rolls-Royce)
             if part in specific_explanations:
                 explanation += specific_explanations[part]
                 matched_explanation = True
                 break
        if not matched_explanation and feature_name.startswith('brand_'):
            brand_name_only = feature_name.split('brand_')[1]
            explanation += f"Keberadaan merek '{brand_name_only}' memiliki pengaruh tertentu pada harga, yang bisa jadi karena persepsi kualitas, target pasar, atau fitur standar yang ditawarkan oleh merek tersebut."
        elif not matched_explanation and (feature_name.startswith('ext_col_') or feature_name.startswith('int_col_')):
            color_name = feature_name.split('_', 2)[-1]
            type_col = "eksterior" if "ext_col" in feature_name else "interior"
            explanation += f"Warna {type_col} '{color_name}' dapat sedikit mempengaruhi preferensi pembeli dan terkadang harga, terutama jika warna tersebut langka atau sedang tren."
        elif not matched_explanation:
             explanation += "Pengaruh fitur ini terhadap harga perlu dipertimbangkan dalam konteks keseluruhan spesifikasi mobil."
    st.markdown(explanation)


# --- Insight & Rekomendasi ---
st.markdown("### 💡 Insight Utama & Rekomendasi")
st.info("""
- **Faktor Utama:** Usia mobil, jarak tempuh, dan tenaga mesin adalah penentu harga yang sangat dominan.
- **Pengaruh Merek:** Merek-merek mewah dan supercar secara signifikan meningkatkan harga karena faktor prestise, kualitas, kustomisasi, dan eksklusivitas.
- **Kondisi & Riwayat:** Mobil dengan riwayat kecelakaan cenderung lebih murah, sementara mobil dengan dokumen bersih (clean title) lebih mahal.
- **Tips:** Gunakan aplikasi ini sebagai panduan awal. Selalu lakukan inspeksi fisik dan bandingkan dengan beberapa sumber sebelum membuat keputusan jual/beli.
""")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© 2025 Prediksi Harga Mobil Bekas | Dibangun oleh [Rizky Wibowo Kusumo] dengan Streamlit</div>", unsafe_allow_html=True)

