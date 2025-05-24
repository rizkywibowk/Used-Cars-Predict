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

df = load_data()

# 2. CLEANING & FEATURE ENGINEERING
def clean_preprocess(df):
    df['price'] = df['price'].astype(str).str.replace('[\$,]', '', regex=True).str.replace(',', '').astype(float)
    df['milage'] = df['milage'].astype(str).str.replace('[\,mi. ]', '', regex=True).replace('-', np.nan)
    df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
    df['car_age'] = 2025 - df['model_year']
    df['engine_hp'] = df['engine'].apply(extract_hp)
    df['engine_L'] = df['engine'].apply(extract_L)
    df['accident_reported'] = df['accident'].apply(lambda x: 1 if 'accident' in str(x).lower() else 0)
    df['has_clean_title'] = df['clean_title'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    for col in ['engine_hp', 'engine_L', 'milage', 'car_age']:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in ['fuel_type', 'ext_col', 'int_col', 'brand', 'transmission']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df = df.dropna(subset=['price'])
    return df

df = clean_preprocess(df)

# 3. SEGMENTASI MESIN
def engine_segment(row):
    engine = str(row['engine']).lower()
    fuel = row['fuel_type'].lower()
    if 'electric' in engine or fuel == 'electric':
        return 'Electric'
    elif 'hybrid' in engine or 'plug-in' in engine or 'hybrid' in fuel:
        return 'Hybrid'
    elif 'v12' in engine or '12 cylinder' in engine:
        return 'V12'
    elif 'v10' in engine or '10 cylinder' in engine:
        return 'V10'
    elif 'v8' in engine or '8 cylinder' in engine:
        return 'V8'
    elif 'v6' in engine or '6 cylinder' in engine:
        return 'V6'
    elif 'i4' in engine or '4 cylinder' in engine or 'flat 4' in engine:
        return 'I4'
    elif 'i3' in engine or '3 cylinder' in engine:
        return 'I3'
    else:
        return 'Other'

df['engine_segment'] = df.apply(engine_segment, axis=1)

# 4. ENCODING FITUR KATEGORIKAL
categorical = ['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'engine_segment']
df_encoded = pd.get_dummies(df, columns=categorical, drop_first=True)

# 5. SPLIT DATA
features = ['car_age', 'milage', 'engine_hp', 'engine_L', 'accident_reported', 'has_clean_title'] + \
           [col for col in df_encoded.columns if any(cat in col for cat in categorical)]
X = df_encoded[features]
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. TRAIN MODEL (sekali saja, lalu simpan)
@st.cache_resource
def train_and_save_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'rf_model.pkl')
    return model

if not st.session_state.get('model_trained', False):
    model = train_and_save_model()
    st.session_state['model_trained'] = True
else:
    model = joblib.load('rf_model.pkl')

# --- STYLING & HEADER
st.markdown("""
    <style>
    .main {background-color: #f2f6fc;}
    .stApp {background-color: #f2f6fc;}
    .title {color: #0a3d62; font-size: 3em; font-weight: bold;}
    .subtitle {color: #3c6382; font-size: 1.3em; margin-bottom: 10px;}
    .footer {text-align: center; color: gray; margin-top: 40px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🚗 Prediksi Harga Mobil Bekas</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aplikasi prediksi harga mobil bekas berbasis Machine Learning. Data lebih dari 400.000 mobil berbagai merek & tipe.</div>', unsafe_allow_html=True)

# --- Sidebar: Eksplorasi Data
st.sidebar.header("🔎 Eksplorasi Data")
if st.sidebar.checkbox("Tampilkan Dataframe"):
    st.dataframe(df.head(100), use_container_width=True)

st.sidebar.subheader("Distribusi Harga")
fig, ax = plt.subplots(figsize=(6,3))
sns.histplot(df['price'], bins=50, ax=ax, color="#0a3d62")
ax.set_xlabel('Harga Mobil')
ax.set_ylabel('Jumlah')
st.sidebar.pyplot(fig)

# --- Main: Form Prediksi
st.markdown("### 📝 Form Prediksi Harga")
with st.form("prediksi_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", sorted(df['brand'].unique()))
        model_in = st.text_input("Model", "")
        model_year = st.slider("Tahun Mobil", int(df['model_year'].min()), int(df['model_year'].max()), 2020)
        milage = st.number_input("Jarak Tempuh (mil)", min_value=0, max_value=500000, value=50000)
    with col2:
        fuel_type = st.selectbox("Tipe Bahan Bakar", sorted(df['fuel_type'].unique()))
        transmission = st.selectbox("Transmisi", sorted(df['transmission'].unique()))
        ext_col = st.selectbox("Warna Eksterior", sorted(df['ext_col'].unique()))
        int_col = st.selectbox("Warna Interior", sorted(df['int_col'].unique()))
        engine = st.text_input("Deskripsi Mesin", "")
        accident_reported = st.selectbox("Pernah Kecelakaan?", ["Tidak", "Ya"])
        has_clean_title = st.selectbox("Clean Title?", ["Tidak", "Ya"])
    submitted = st.form_submit_button("Prediksi Harga 🚙")

if submitted:
    car_age = 2025 - model_year
    engine_hp = extract_hp(engine)
    engine_L = extract_L(engine)
    accident_reported_bin = 1 if accident_reported == "Ya" else 0
    has_clean_title_bin = 1 if has_clean_title == "Ya" else 0
    engine_seg = engine_segment({'engine': engine, 'fuel_type': fuel_type})
    input_dict = {
        'car_age': car_age,
        'milage': milage,
        'engine_hp': engine_hp if not np.isnan(engine_hp) else df['engine_hp'].mean(),
        'engine_L': engine_L if not np.isnan(engine_L) else df['engine_L'].mean(),
        'accident_reported': accident_reported_bin,
        'has_clean_title': has_clean_title_bin,
    }
    for col in features:
        if col.startswith('brand_'):
            input_dict[col] = 1 if col == f'brand_{brand}' else 0
        elif col.startswith('fuel_type_'):
            input_dict[col] = 1 if col == f'fuel_type_{fuel_type}' else 0
        elif col.startswith('transmission_'):
            input_dict[col] = 1 if col == f'transmission_{transmission}' else 0
        elif col.startswith('ext_col_'):
            input_dict[col] = 1 if col == f'ext_col_{ext_col}' else 0
        elif col.startswith('int_col_'):
            input_dict[col] = 1 if col == f'int_col_{int_col}' else 0
        elif col.startswith('engine_segment_'):
            input_dict[col] = 1 if col == f'engine_segment_{engine_seg}' else 0
    for col in features:
        if col not in input_dict:
            input_dict[col] = 0
    input_df = pd.DataFrame([input_dict])
    pred = model.predict(input_df)
    st.success(f"💰 Estimasi harga mobil bekas Anda: **${pred[0]:,.2f}**")

# --- Feature Importance
st.markdown("### 🌟 Feature Importance")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax2, palette="Blues_r")
ax2.set_xlabel('Tingkat Pengaruh')
ax2.set_ylabel('Fitur')
ax2.set_title('10 Fitur Paling Berpengaruh')
st.pyplot(fig2)

st.markdown("#### Penjelasan Angka pada Fitur Importance:")
for feature, importance in feat_imp.items():
    st.markdown(f"- **{feature}**: Memberikan kontribusi sebesar **{importance:.4f}** terhadap prediksi harga mobil. Semakin tinggi nilainya, semakin besar pengaruh fitur tersebut dalam menentukan harga.")

# --- Insight
st.markdown("### 💡 Insight & Rekomendasi")
st.info("""
- Mobil lebih muda, milage rendah, dan mesin bertenaga tinggi cenderung lebih mahal.
- Brand premium memiliki harga rata-rata lebih tinggi.
- Gunakan aplikasi ini untuk estimasi harga sebelum membeli/menjual mobil bekas.
""")

# --- Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© 2025 Prediksi Harga Mobil Bekas | Dibuat dengan ❤️ menggunakan Streamlit</div>", unsafe_allow_html=True)
