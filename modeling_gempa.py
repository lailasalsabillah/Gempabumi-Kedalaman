# modeling_gempa.py
# Utility untuk memuat model & melakukan prediksi kedalaman gempa

import os
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------
# Konfigurasi lokasi file model
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
XGB_PATH = os.path.join(MODELS_DIR, "xgb_depth_class.pkl")
LSTM_PATH = os.path.join(MODELS_DIR, "lstm_depth_class.keras")

# ---------------------------------------------------------
# Peta label & tingkat bahaya
# ---------------------------------------------------------
LABEL_MAP = {
    0: "Shallow (< 70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (> 300 km)",
}

DANGER_MAP = {
    0: "Bahaya Tinggi",
    1: "Bahaya Sedang",
    2: "Bahaya Rendah",
}

FEATURE_ORDER = [
    "year",
    "latitude",
    "longitude",
    "mag",
    "gap",
    "dmin",
    "rms",
    "horizontalError",
    "depthError",
    "magError",
]

# ---------------------------------------------------------
# Muat XGBoost & Scaler
# ---------------------------------------------------------
try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None

try:
    xgb_model = joblib.load(XGB_PATH)
except Exception:
    xgb_model = None

# ---------------------------------------------------------
# Muat LSTM (opsional – bisa gagal di Streamlit Cloud)
# ---------------------------------------------------------
HAS_TF = False
lstm_model = None

try:
    from tensorflow.keras.models import load_model  # type: ignore

    if os.path.exists(LSTM_PATH):
        lstm_model = load_model(LSTM_PATH)
        HAS_TF = True
except Exception:
    # Di server (Python 3.13) TensorFlow mungkin tidak tersedia.
    HAS_TF = False
    lstm_model = None


# ---------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------
def _to_dataframe(features: dict) -> pd.DataFrame:
    """
    Mengubah dict input menjadi DataFrame 1 baris
    dengan urutan kolom yang sama seperti saat training.
    """
    data = {k: features.get(k) for k in FEATURE_ORDER}
    return pd.DataFrame([data])


# ---------------------------------------------------------
# Fungsi utama untuk Streamlit
# ---------------------------------------------------------
def predict_depth_class(features: dict) -> dict:
    """
    Parameters
    ----------
    features : dict
        {
          "year": int,
          "latitude": float,
          "longitude": float,
          "mag": float,
          "gap": float,
          "dmin": float,
          "rms": float,
          "horizontalError": float,
          "depthError": float,
          "magError": float
        }

    Returns
    -------
    dict dengan dua kunci: "XGBoost" dan "LSTM"
    """

    df = _to_dataframe(features)

    results = {}

    # ===========================
    # 1. Prediksi XGBoost
    # ===========================
    if xgb_model is not None:
        cls_xgb = int(xgb_model.predict(df)[0])
        results["XGBoost"] = {
            "class": cls_xgb,
            "label": LABEL_MAP.get(cls_xgb, "Tidak diketahui"),
            "danger": DANGER_MAP.get(cls_xgb, "-"),
        }
    else:
        results["XGBoost"] = {
            "class": None,
            "label": "Model XGBoost tidak ditemukan",
            "danger": "-",
        }

    # ===========================
    # 2. Prediksi LSTM (jika ada)
    # ===========================
    if HAS_TF and lstm_model is not None and scaler is not None:
        x_scaled = scaler.transform(df.values)  # (1, 10)
        x_lstm = x_scaled.reshape((1, 1, x_scaled.shape[1]))  # (1, 1, 10)

        proba = lstm_model.predict(x_lstm, verbose=0)[0]
        cls_lstm = int(np.argmax(proba))

        results["LSTM"] = {
            "class": cls_lstm,
            "label": LABEL_MAP.get(cls_lstm, "Tidak diketahui"),
            "danger": DANGER_MAP.get(cls_lstm, "-"),
        }
    else:
        # Graceful fallback kalau TensorFlow tidak bisa di-install di server
        results["LSTM"] = {
            "class": None,
            "label": "Model LSTM tidak tersedia di server",
            "danger": "-",
        }

    return results
