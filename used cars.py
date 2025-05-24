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
    df = pd.read_csv('used_cars.csv')
    return df

df_original = load_data() # Simpan original untuk selectbox
df = df_original.copy() # Bekerja dengan copy

# 2. CLEANING & FEATURE ENGINEERING
def clean_preprocess(df_to_process):
    # Clean price
    df_to_process['price'] = df_to_process['price'].astype(str).str.replace('[\$,]', '', regex=True).str.replace(',', '').astype(float)
    # Clean milage
    df_to_process['milage'] = df_to_process['milage'].astype(str).str.replace('[\,mi. ]', '', regex=True).replace('-', np.nan)
    df_to_process['milage'] = pd.to_numeric(df_to_process['milage'], errors='coerce')
    # Car age
    current_year = pd.Timestamp.now().year # Lebih dinamis
    df_to_process['car_age'] = current_year - df_to_process['model_year']
    # Clean engine_hp & engine_L
    df_to_process['engine_hp'] = df_to_process['engine'].apply(extract_hp)
    df_to_process['engine_L'] = df_to_process['engine'].apply(extract_L)
    # Clean accident_reported & has_clean_title
    df_to_process['accident_reported'] = df_to_process['accident'].apply(lambda x: 1 if 'accident' in str(x).lower() else 0)
    df_to_process['has_clean_title'] = df_to_process['clean_title'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    # Handle missing values
    for col in ['engine_hp', 'engine_L', 'milage', 'car_age']:
        df_to_process[col].fillna(df_to_process[col].mean(), inplace=True)
    for col in ['fuel_type', 'ext_col', 'int_col', 'brand', 'transmission']:
        if col in df_to_process.columns: # Pastikan kolom ada
            df_to_process[col].fillna(df_to_process[col].mode()[0], inplace=True)
    df_to_process = df_to_process.dropna(subset=['price'])
    return df_to_process

df = clean_preprocess(df)

# 3. SEGMENTASI MESIN
def engine_segment(row):
    engine_text = str(row.get('engine', '')).lower() # Ambil dari kolom 'engine'
    fuel = str(row.get('fuel_type', '')).lower() # Ambil dari kolom 'fuel_type'
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
# Pastikan semua kolom ada sebelum encoding
existing_categorical_cols = [col for col in categorical_cols_to_encode if col in df.columns]
df_encoded = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True)


# 5. SPLIT DATA
# Daftar fitur dasar + kolom hasil one-hot encoding
base_features = ['car_age', 'milage', 'engine_hp', 'engine_L', 'accident_reported', 'has_clean_title']
encoded_feature_cols = [col for col in df_encoded.columns if any(cat_col in col for cat_col in existing_categorical_cols)]
final_features = base_features + encoded_feature_cols

# Pastikan semua fitur ada di df_encoded sebelum membuat X
X = df_encoded[[feat for feat in final_features if feat in df_encoded.columns]]
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. TRAIN MODEL (sekali saja, lalu simpan)
@st.cache_resource
def train_and_save_model(X_train_data, y_train_data):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 untuk memakai semua core
    model.fit(X_train_data, y_train_data)
    joblib.dump(model, 'rf_model.pkl')
    return model

# Panggil fungsi training hanya jika model belum di-cache
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
    st.dataframe(df_original.head(100), use_container_width=True) # Tampilkan df original

st.sidebar.subheader("Distribusi Harga Mobil Bekas")
fig_hist, ax_hist = plt.subplots(figsize=(7,4)) # Ukuran disesuaikan
sns.histplot(df['price'], bins=50, ax=ax_hist, color="#0a3d62", kde=True)
ax_hist.set_title('Distribusi Harga', fontsize=14)
ax_hist.set_xlabel('Harga Mobil (USD)', fontsize=12)
ax_hist.set_ylabel('Jumlah Mobil', fontsize=12)
ax_hist.tick_params(axis='x', labelsize=10)
ax_hist.tick_params(axis='y', labelsize=10)
plt.tight_layout() # Agar tidak terpotong
st.sidebar.pyplot(fig_hist)

# --- Main: Form Prediksi ---
st.markdown("### 📝 Masukkan Spesifikasi Mobil Anda")
with st.form("prediksi_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand_options = sorted(df_original['brand'].unique()) # Dari df original
        brand = st.selectbox("🚗 Merek Mobil", brand_options, help="Pilih merek mobil.")
        
        model_input_key = "model_input_text" # Kunci unik untuk text_input
        model_in = st.text_input("모델 Model Mobil", key=model_input_key, help="Masukkan model mobil, contoh: Camry, Civic, F-150.")
        
        min_year = int(df_original['model_year'].min())
        max_year = int(df_original['model_year'].max())
        model_year = st.slider("📅 Tahun Pembuatan", min_year, max_year, 2020, help=f"Pilih tahun pembuatan mobil, antara {min_year} dan {max_year}.")
        
        milage = st.number_input("🛣️ Jarak Tempuh (mil)", min_value=0, max_value=int(df_original['milage'].max() + 50000), value=50000, step=1000, help="Masukkan total jarak tempuh mobil dalam mil.")

    with col2:
        fuel_type_options = sorted(df_original['fuel_type'].unique())
        fuel_type = st.selectbox("⛽ Tipe Bahan Bakar", fuel_type_options, help="Pilih tipe bahan bakar mobil.")
        
        transmission_options = sorted(df_original['transmission'].unique())
        transmission = st.selectbox("⚙️ Transmisi", transmission_options, help="Pilih jenis transmisi mobil.")
        
        engine = st.text_input("🛠️ Deskripsi Mesin (Contoh: 300.0HP 3.7L V6 Cylinder Engine)", help="Masukkan deskripsi mesin, contoh: 2.0L I4 Turbo.")
        
        accident_reported = st.selectbox("💥 Riwayat Kecelakaan?", ["Tidak", "Ya"], help="Apakah mobil pernah dilaporkan mengalami kecelakaan?")
        
        has_clean_title = st.selectbox("📜 Status Dokumen (Clean Title)?", ["Ya", "Tidak"], help="Apakah mobil memiliki dokumen yang bersih (clean title)?")

    # Tombol submit di tengah
    submit_col_left, submit_col_center, submit_col_right = st.columns([2,3,2])
    with submit_col_center:
        submitted = st.form_submit_button("✨ Prediksi Harga Sekarang!")

if submitted:
    with st.spinner("🔍 Menganalisis dan Memprediksi Harga..."):
        car_age_input = current_year - model_year
        engine_hp_input = extract_hp(engine)
        engine_L_input = extract_L(engine)
        accident_reported_bin_input = 1 if accident_reported == "Ya" else 0
        has_clean_title_bin_input = 1 if has_clean_title == "Ya" else 0
        
        # Untuk engine_segment, kita perlu data 'engine' dan 'fuel_type' dari input
        engine_seg_input_data = {'engine': engine, 'fuel_type': fuel_type}
        engine_seg_input = engine_segment(engine_seg_input_data)

        input_dict = {
            'car_age': car_age_input,
            'milage': milage,
            'engine_hp': engine_hp_input if not np.isnan(engine_hp_input) else X_train['engine_hp'].mean(), # Ambil mean dari X_train
            'engine_L': engine_L_input if not np.isnan(engine_L_input) else X_train['engine_L'].mean(),   # Ambil mean dari X_train
            'accident_reported': accident_reported_bin_input,
            'has_clean_title': has_clean_title_bin_input,
        }
        
        # One-hot encoding manual untuk input user
        for cat_col_original in existing_categorical_cols: # Iterasi berdasarkan kolom kategori asli
            # Dapatkan nilai input user untuk kolom kategori ini
            user_value_for_category = locals().get(cat_col_original.replace('_segment',''), locals().get(cat_col_original)) # Handle 'engine_segment' vs 'engine_seg_input'
            if cat_col_original == 'engine_segment': user_value_for_category = engine_seg_input

            # Cari semua kolom one-hot yang berawal dari nama kolom kategori ini
            for encoded_col_name in X_train.columns: # Gunakan kolom dari X_train untuk konsistensi
                if encoded_col_name.startswith(f"{cat_col_original}_"):
                    # Ambil suffix dari nama kolom one-hot (yaitu nilai kategorinya)
                    category_value_in_col_name = encoded_col_name.split(f"{cat_col_original}_", 1)[1]
                    # Jika nilai input user cocok dengan nilai kategori di nama kolom, set 1, else 0
                    input_dict[encoded_col_name] = 1 if user_value_for_category == category_value_in_col_name else 0
        
        # Pastikan semua kolom fitur yang ada di X_train ada di input_dict, jika tidak, set 0
        for col_in_model in X_train.columns:
            if col_in_model not in input_dict:
                if col_in_model not in base_features: # Hanya untuk kolom one-hot yang mungkin tidak ter-set
                     input_dict[col_in_model] = 0

        input_df = pd.DataFrame([input_dict])[X_train.columns] # Pastikan urutan kolom sama dengan X_train
        
        pred = model.predict(input_df)
        
        st.markdown(f"<div class='stSuccess'>💰 Estimasi harga mobil bekas Anda: <strong>${pred[0]:,.2f}</strong></div>", unsafe_allow_html=True)

# --- Feature Importance ---
st.markdown("### 🌟 Faktor Penentu Harga Mobil (Feature Importance)")
importances = model.feature_importances_
# Gunakan nama fitur dari X_train.columns untuk konsistensi
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(10)

fig_imp, ax_imp = plt.subplots(figsize=(10,6)) # Ukuran disesuaikan
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax_imp, palette="viridis") # Ganti palette
ax_imp.set_title('10 Faktor Paling Berpengaruh Terhadap Harga', fontsize=15)
ax_imp.set_xlabel('Tingkat Pengaruh (Feature Importance Score)', fontsize=12)
ax_imp.set_ylabel('Faktor (Fitur)', fontsize=12)
ax_imp.tick_params(axis='x', labelsize=10)
ax_imp.tick_params(axis='y', labelsize=10)
plt.tight_layout()
st.pyplot(fig_imp)

st.markdown("#### Penjelasan Mendalam Setiap Faktor:")
# Buat dictionary untuk penjelasan yang lebih spesifik
specific_explanations = {
    "milage": "Jarak tempuh yang lebih tinggi umumnya mengindikasikan penggunaan yang lebih intensif dan potensi keausan, sehingga cenderung menurunkan harga jual kembali mobil.",
    "car_age": "Usia mobil adalah faktor depresiasi utama. Semakin tua mobil, nilainya cenderung semakin menurun karena perkembangan teknologi baru dan kondisi fisik yang mungkin menurun.",
    "engine_hp": "Tenaga mesin (Horsepower) seringkali berbanding lurus dengan performa dan segmen mobil. Mobil dengan HP lebih tinggi biasanya lebih mahal karena ditujukan untuk pasar performa atau kemewahan.",
    "engine_L": "Kapasitas mesin (Liter) juga berkorelasi dengan tenaga dan harga. Mesin dengan volume lebih besar sering ditemukan pada mobil mewah atau SUV besar, yang harganya lebih tinggi.",
    "brand_Rolls-Royce": "Merek Rolls-Royce adalah simbol status dan kemewahan tertinggi. Setiap unit seringkali adalah 'bespoke' atau dibuat khusus sesuai pesanan (made-by-order), menggunakan material terbaik, dan dirakit tangan oleh pengrajin ahli. Ini menciptakan nilai eksklusivitas dan seni yang sangat tinggi, menjadikan harganya jauh di atas rata-rata [2][3][4][5][6].",
    "brand_Lamborghini": "Merek Lamborghini identik dengan supercar berperforma ekstrim, desain avant-garde, dan produksi terbatas. Eksklusivitas dan teknologi canggihnya membuat harganya sangat tinggi.",
    "brand_Ferrari": "Ferrari adalah ikon mobil sport mewah dengan sejarah balap yang kaya. Setiap modelnya menawarkan performa tinggi, desain menawan, dan status prestisius, yang berkontribusi pada nilai jualnya.",
    "brand_Porsche": "Porsche terkenal dengan rekayasa presisi Jerman, mobil sport yang ikonik (seperti 911), dan keseimbangan antara performa di trek dan kenyamanan harian. Reputasi merek dan kualitasnya menjaga nilai mobil tetap tinggi.",
    "engine_segment_V12": "Mesin V12 adalah konfigurasi mesin yang sangat bertenaga dan kompleks, biasanya hanya ditemukan pada mobil sport paling eksotis atau sedan ultra-mewah. Kelangkaan dan performanya membuat mobil dengan mesin ini sangat mahal.",
    "has_clean_title": "Status 'Clean Title' menunjukkan bahwa mobil tidak memiliki riwayat masalah legal atau kerusakan parah (seperti salvage title). Ini memberikan rasa aman bagi pembeli dan secara signifikan meningkatkan nilai jual mobil.",
    "accident_reported": "Riwayat kecelakaan yang dilaporkan, terutama jika parah, dapat secara signifikan menurunkan nilai mobil karena potensi kerusakan struktural atau masalah tersembunyi."
}

for feature_name, importance_score in feat_imp.items():
    # Base explanation
    explanation = f"- **{feature_name}**: Memberikan kontribusi sebesar **{importance_score:.4f}** dalam model prediksi harga. "
    
    # Add specific explanation if available
    clean_feature_name = feature_name.split('_', 1)[1] if feature_name.startswith('brand_') or feature_name.startswith('engine_segment_') or feature_name.startswith('ext_col_') or feature_name.startswith('int_col_') or feature_name.startswith('fuel_type_') or feature_name.startswith('transmission_') else feature_name
    # Adjust for brand_Rolls-Royce case specifically
    if "brand_Rolls-Royce" in feature_name: clean_feature_name = "brand_Rolls-Royce"


    if clean_feature_name in specific_explanations:
        explanation += specific_explanations[clean_feature_name]
    elif feature_name.startswith('brand_'):
        brand_name_only = feature_name.split('brand_')[1]
        explanation += f"Keberadaan merek '{brand_name_only}' memiliki pengaruh tertentu pada harga, yang bisa jadi karena persepsi kualitas, target pasar, atau fitur standar yang ditawarkan oleh merek tersebut."
    elif feature_name.startswith('ext_col_') or feature_name.startswith('int_col_'):
        color_name = feature_name.split('_', 2)[-1]
        type_col = "eksterior" if "ext_col" in feature_name else "interior"
        explanation += f"Warna {type_col} '{color_name}' dapat sedikit mempengaruhi preferensi pembeli dan terkadang harga, terutama jika warna tersebut langka atau sedang tren."
    else:
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
st.markdown("<div class='footer'>© 2025 Prediksi Harga Mobil Bekas | Dibangun oleh [Nama Anda/Tim Anda] dengan Streamlit</div>", unsafe_allow_html=True)

