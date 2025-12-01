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
    page_icon="🌋"
)

# ----------------------------------------
# CSS Background
# ----------------------------------------
bg_home = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1502126324834-38f0401b37b0?auto=format&fit=crop&w=1350&q=80');
    background-size: cover;
    background-position: center;
}
</style>
"""

bg_white = """
<style>
[data-testid="stAppViewContainer"] {
    background: #ffffff !important;
}
</style>
"""

# ----------------------------------------
# CSS Card
# ----------------------------------------
st.markdown("""
<style>
.result-card {
    background: #f8f9fa;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e1e1e1;
    margin-bottom: 20px;
    box-shadow: 1px 2px 6px rgba(0,0,0,0.08);
}
.title-text {
    font-size: 28px;
    font-weight: 700;
    color: #333;
}
.pred-label {
    font-size: 22px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

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
if st.sidebar.button("🔍 Prediksi Gempa"):
    st.session_state.page = "predict"


# ============================================================
# PAGE 1: HOME
# ============================================================
if st.session_state.page == "home":
    st.markdown(bg_home, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: rgba(0,0,0,0.55);
        padding: 45px;
        border-radius: 20px;
        margin-top: 120px;
        text-align: center;">
        <h1 style="color:white; font-size:50px;">
            🌋 Prediksi Kedalaman Gempa Bumi
        </h1>
        <p style="color:white; font-size:20px; margin-top:15px;">
            Sistem prediksi berbasis model <b>LSTM</b> dan <b>XGBoost</b> 
            untuk mengklasifikasikan kedalaman gempa bumi menjadi
            <i>shallow</i>, <i>intermediate</i>, dan <i>deep</i>.
        </p>
        <p style="color:white; font-size:18px;">
            Klik menu di sidebar untuk mulai memprediksi.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 2: PREDICTION
# ============================================================
elif st.session_state.page == "predict":
    st.markdown(bg_white, unsafe_allow_html=True)

    st.title("🔍 Prediksi Kedalaman Gempa Bumi")
    st.write("Silakan masukkan parameter gempa di bawah ini:")

    col1, col2 = st.columns(2)

    # -----------------------------------
    # Input Slider
    # -----------------------------------
    with col1:
        year = st.slider("Tahun Kejadian", 2020, 2024, 2023)
        latitude = st.slider("Latitude", -12.0, 6.0, 0.0, step=0.01)
        longitude = st.slider("Longitude", 95.0, 141.0, 120.0, step=0.01)
        mag = st.slider("Magnitude (M)", 3.0, 8.0, 5.0)

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
            "magError": magError
        }

        result = predict_depth_class(features)

        st.subheader("📌 Hasil Prediksi Model")

        # ------------------ XGBoost ------------------
        xgb = result["XGBoost"]
        st.markdown(f"""
        <div class="result-card">
            <div class="title-text">🤖 XGBoost</div>
            <p class="pred-label">Kelas: <b>{xgb['label']}</b></p>
            <p style="color:{xgb['color']}; font-size:20px;">
                Tingkat Bahaya: <b>{xgb['danger']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ------------------ LSTM ------------------
        lstm = result["LSTM"]
        st.markdown(f"""
        <div class="result-card">
            <div class="title-text">🧠 LSTM</div>
            <p class="pred-label">Kelas: <b>{lstm['label']}</b></p>
            <p style="color:{lstm['color']}; font-size:20px;">
                Tingkat Bahaya: <b>{lstm['danger']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Tombol kembali
    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
