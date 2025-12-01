# ============================================================
# app.py - Aplikasi Streamlit Prediksi Kedalaman Gempa Bumi
# ============================================================

import streamlit as st
from modeling_gempa import predict_depth_class

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Prediksi Kedalaman Gempa",
    layout="wide",
    page_icon="🌋",
)

# ----------------------------------------
# CSS sederhana (opsional)
# ----------------------------------------
BG_HOME = """
<style>
.main-block {
    background: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 1rem;
}
</style>
"""

BG_WHITE = """
<style>
</style>
"""

st.markdown(BG_HOME, unsafe_allow_html=True)

# ----------------------------------------
# STATE PAGE
# ----------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ----------------------------------------
# SIDEBAR
# ----------------------------------------
st.sidebar.title("Menu Navigasi")
if st.sidebar.button("🏠 Beranda"):
    st.session_state.page = "home"
if st.sidebar.button("📊 Prediksi Gempa"):
    st.session_state.page = "predict"

# ============================================================
# PAGE 1: HOME
# ============================================================
if st.session_state.page == "home":
    st.markdown(
        """
        <div class="main-block">
            <h1>🌋 Prediksi Kedalaman Gempa Bumi</h1>
            <p>
            Sistem klasifikasi kedalaman gempa bumi menggunakan algoritma
            <b>LSTM</b> dan <b>XGBoost</b> berdasarkan data gempa BMKG periode 2020–2024.
            </p>
            <p>
            Gunakan menu di sidebar untuk membuka halaman <b>Prediksi Gempa</b>,
            lalu masukkan parameter kejadian gempa untuk mengetahui apakah
            hiposenternya dangkal, menengah, atau dalam.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PAGE 2: PREDICTION
# ============================================================
elif st.session_state.page == "predict":
    st.markdown(BG_WHITE, unsafe_allow_html=True)

    st.title("📊 Prediksi Kedalaman Gempa Bumi")
    st.write("Silakan masukkan parameter gempa di bawah ini:")

    col1, col2 = st.columns(2)

    # -----------------------------------
    # Input Slider
    # -----------------------------------
    with col1:
        year = st.slider("Tahun Kejadian", 2020, 2024, 2023)
        latitude = st.slider("Latitude", -12.0, 6.0, 0.0, step=0.01)
        longitude = st.slider("Longitude", 95.0, 141.0, 120.0, step=0.01)
        mag = st.slider("Magnitudo (M)", 3.0, 8.0, 5.0)

    with col2:
        gap = st.slider("Gap", 10, 300, 80)
        dmin = st.slider("Dmin (km)", 0.0, 30.0, 2.0)
        rms = st.slider("RMS", 0.0, 2.0, 0.7)
        horizontalError = st.slider("Horizontal Error", 1.0, 25.0, 8.0)
        depthError = st.slider("Depth Error", 0.5, 30.0, 5.0)
        magError = st.slider("Magnitude Error", 0.02, 1.0, 0.1)

    # -----------------------------------
    # Tombol Prediksi
    # -----------------------------------
    if st.button("⚡ Prediksi Sekarang"):
        features = {
            "year": year,
            "latitude": latitude,
            "longitude": longitude,
            "mag": mag,
            "gap": gap,
            "dmin": dmin,
            "rms": rms,
            "horizontalError": horizontalError,
            "depthError": depthError,
            "magError": magError,
        }

        result = predict_depth_class(features)

        st.subheader("🔍 Hasil Prediksi Model")

        # ------------------ XGBoost ------------------
        xgb = result.get("XGBoost", {})
        st.markdown(
            f"""
            **XGBoost**
            - Kelas Kedalaman : `{xgb.get('label', '-')}`
            - Tingkat Bahaya  : `{xgb.get('danger', '-')}`
            """,
        )

        # ------------------ LSTM ------------------
        lstm = result.get("LSTM", {})
        st.markdown(
            f"""
            **LSTM**
            - Kelas Kedalaman : `{lstm.get('label', '-')}`
            - Tingkat Bahaya  : `{lstm.get('danger', '-')}`
            """,
        )

    # Tombol kembali
    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
