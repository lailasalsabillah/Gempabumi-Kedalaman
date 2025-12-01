# ============================================================
# STREAMLIT - KLASIFIKASI KEDALAMAN GEMPA
# Menggunakan LSTM & XGBoost
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Kedalaman Gempa",
    layout="wide",
    page_icon="🌋"
)

st.title("🌋 Prediksi Kelas Kedalaman Gempa Bumi")
st.write("Model menggunakan **LSTM** & **XGBoost** yang dilatih pada data 2020–2024.")

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")
lstm_model = load_model("models/lstm_depth_class.keras")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

danger_map = {
    0: ("Bahaya Tinggi", "red"),
    1: ("Bahaya Sedang", "orange"),
    2: ("Bahaya Rendah", "green")
}

# ------------------------------------------------------------
# INPUT FEATURES
# ------------------------------------------------------------
st.subheader("🔧 Input Parameter Gempa")

col1, col2 = st.columns(2)

with col1:
    year = st.slider("Tahun Kejadian", 2020, 2024, 2023)
    latitude = st.slider("Latitude", -12.0, 6.0, 0.0, step=0.01)
    longitude = st.slider("Longitude", 95.0, 141.0, 120.0, step=0.01)
    mag = st.slider("Magnitude", 3.0, 8.0, 5.0)

with col2:
    gap = st.slider("Gap", 10, 300, 80)
    dmin = st.slider("Dmin", 0.0, 30.0, 2.0)
    rms = st.slider("RMS", 0.0, 2.0, 0.7)
    horizontalError = st.slider("Horizontal Error", 1.0, 25.0, 8.0)
    depthError = st.slider("Depth Error", 0.5, 30.0, 5.0)
    magError = st.slider("Magnitude Error", 0.02, 1.0, 0.1)

# ------------------------------------------------------------
# PREDIKSI
# ------------------------------------------------------------
input_data = np.array([[year, latitude, longitude, mag, gap, dmin, rms,
                        horizontalError, depthError, magError]])

scaled_data = scaler.transform(input_data)

# XGBoost Prediction
pred_xgb = xgb_model.predict(input_data)[0]

# LSTM Prediction
lstm_input = scaled_data.reshape(1, 1, scaled_data.shape[1])
pred_lstm = np.argmax(lstm_model.predict(lstm_input), axis=1)[0]

# ------------------------------------------------------------
# TOMBOL PREDIKSI
# ------------------------------------------------------------
if st.button("🔍 Prediksi Kedalaman"):
    st.subheader("Hasil Prediksi")

    col3, col4 = st.columns(2)

    # XGBoost
    with col3:
        kelas = pred_xgb
        label = label_map[kelas]
        danger, color = danger_map[kelas]
        st.markdown(f"### 🤖 XGBoost")
        st.markdown(f"**Kelas Kedalaman:** {label}")
        st.markdown(f"**Tingkat Bahaya:** <span style='color:{color};'>{danger}</span>", unsafe_allow_html=True)

    # LSTM
    with col4:
        kelas2 = pred_lstm
        label2 = label_map[kelas2]
        danger2, color2 = danger_map[kelas2]
        st.markdown(f"### 🧠 LSTM")
        st.markdown(f"**Kelas Kedalaman:** {label2}")
        st.markdown(f"**Tingkat Bahaya:** <span style='color:{color2};'>{danger2}</span>", unsafe_allow_html=True)

