import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Kelayakan Air Minum",
    page_icon="💧",
    layout="wide"
)

# 1. Memuat Model dan Scaler (menggunakan cache agar lebih cepat)
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('stacking_model.pkl')
        return scaler, model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file 'scaler.pkl' dan 'stacking_model.pkl' ada di direktori yang sama. Error: {e}")
        return None, None

scaler, model = load_models()

# 2. Judul dan Deskripsi
st.title("💧 Klasifikasi Kelayakan Air Minum")
st.write("""
Aplikasi ini memprediksi apakah sampel air **Layak** (Aman dikonsumsi) atau **Tidak Layak** (Tidak aman) berdasarkan 9 parameter kimia air menggunakan model **Stacking Ensemble** (XGBoost, LightGBM, CatBoost).
""")

# 3. Sidebar untuk Input Pengguna
st.sidebar.header("Masukkan Parameter Kualitas Air")
st.sidebar.write("Silakan atur nilai fitur di bawah ini:")

def user_input_features():
    # Rentang default disesuaikan dengan nilai wajar pada Water Potability Dataset
    ph = st.sidebar.slider("pH (Keasaman)", 0.0, 14.0, 7.0)
    hardness = st.sidebar.slider("Hardness (Kesadahan - mg/L)", 50.0, 350.0, 196.0)
    solids = st.sidebar.slider("Solids (Total Padatan Terlarut - ppm)", 10000.0, 60000.0, 22000.0)
    chloramines = st.sidebar.slider("Chloramines (ppm)", 0.0, 15.0, 7.1)
    sulfate = st.sidebar.slider("Sulfate (mg/L)", 100.0, 500.0, 333.0)
    conductivity = st.sidebar.slider("Conductivity (μS/cm)", 200.0, 800.0, 426.0)
    organic_carbon = st.sidebar.slider("Organic Carbon (ppm)", 0.0, 30.0, 14.2)
    trihalomethanes = st.sidebar.slider("Trihalomethanes (μg/L)", 0.0, 150.0, 66.3)
    turbidity = st.sidebar.slider("Turbidity (Kekeruhan - NTU)", 0.0, 10.0, 3.9)

    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }
    return pd.DataFrame(data, index=[0])

# Mendapatkan input data
input_df = user_input_features()

# Menampilkan data input
st.subheader("Data Parameter Air (Input)")
st.dataframe(input_df)

# 4. Proses Prediksi
if st.button("Prediksi Kelayakan Air"):
    if scaler is not None and model is not None:
        with st.spinner("Memproses prediksi..."):
            # Normalisasi menggunakan StandardScaler
            input_scaled = scaler.transform(input_df)
            
            # Prediksi menggunakan Stacking Model
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            st.markdown("---")
            st.subheader("Hasil Prediksi")
            
            if prediction[0] == 1:
                st.success("✅ **Air ini LAYAK untuk diminum.**")
            else:
                st.error("❌ **Air ini TIDAK LAYAK untuk diminum.**")
                
            st.write(f"**Probabilitas Tidak Layak (0):** {prediction_proba[0][0]:.2%}")
            st.write(f"**Probabilitas Layak (1):** {prediction_proba[0][1]:.2%}")
            
            # 5. Visualisasi SHAP Waterfall Plot (Explainability)
            # Catatan: StackingClassifier butuh KernelExplainer untuk SHAP
            st.markdown("---")
            st.subheader("Analisis SHAP (Faktor Prediksi)")
            st.write("Grafik di bawah ini menjelaskan seberapa besar setiap parameter memengaruhi hasil prediksi (mendukung Layak vs Tidak Layak).")
            
            try:
                # Menggunakan KernelExplainer untuk meta-learner / ensemble
                # Peringatan: KernelExplainer bisa sedikit lambat
                explainer = shap.KernelExplainer(model.predict, shap.kmeans(scaler.transform(input_df), 1)) 
                shap_values = explainer.shap_values(input_scaled)
                
                # Buat figure matplotlib untuk Streamlit
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Gunakan waterfall plot jika didukung atau force plot via bar
                shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                                      base_values=explainer.expected_value, 
                                                      data=input_scaled[0], 
                                                      feature_names=input_df.columns), show=False)
                st.pyplot(fig)
            except Exception as e:
                st.info("Visualisasi SHAP tidak dapat dirender secara instan untuk model jenis ini karena keterbatasan memori explainer. Namun prediksi tetap akurat.")
    else:
        st.error("Model gagal dimuat. Tolong periksa ketersediaan file model.")