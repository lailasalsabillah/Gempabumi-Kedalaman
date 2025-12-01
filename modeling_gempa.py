# =============================================================
# modeling_gempa.py
# Modul untuk prediksi kedalaman gempa menggunakan LSTM & XGBoost
# =============================================================

import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------------------------------------------------
# LABEL DAN TINGKAT BAHAYA
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# LOAD MODEL (dipanggil sekali)
# -------------------------------------------------------------
def load_models():
    """Load scaler, XGBoost, dan LSTM."""
    scaler = joblib.load("models/scaler.pkl")
    xgb_model = joblib.load("models/xgb_depth_class.pkl")
    lstm_model = load_model("models/lstm_depth_class.keras")
    return scaler, xgb_model, lstm_model


# -------------------------------------------------------------
# PREDIKSI
# -------------------------------------------------------------
def predict_depth_class(features_dict):
    """
    Menerima input berupa dictionary berisi fitur.
    Contoh:
    {
        'year': 2023,
        'latitude': -6.3,
        'longitude': 120.1,
        'mag': 5.0,
        'gap': 80,
        'dmin': 2.5,
        'rms': 0.7,
        'horizontalError': 8.0,
        'depthError': 5.0,
        'magError': 0.1
    }
    """

    # Urutan fitur
    feature_order = [
        "year", "latitude", "longitude", "mag",
        "gap", "dmin", "rms",
        "horizontalError", "depthError", "magError"
    ]

    # Ubah ke array
    input_data = np.array([[features_dict[f] for f in feature_order]])

    # Load model
    scaler, xgb_model, lstm_model = load_models()

    # Scaling
    scaled_data = scaler.transform(input_data)

    # XGBoost prediksi
    pred_xgb = int(xgb_model.predict(input_data)[0])

    # LSTM input reshape (1,1,features)
    lstm_input = scaled_data.reshape(1, 1, scaled_data.shape[1])
    pred_lstm = int(np.argmax(lstm_model.predict(lstm_input), axis=1)[0])

    # Return hasil lengkap
    return {
        "XGBoost": {
            "class": pred_xgb,
            "label": label_map[pred_xgb],
            "danger": danger_map[pred_xgb][0],
            "color": danger_map[pred_xgb][1]
        },
        "LSTM": {
            "class": pred_lstm,
            "label": label_map[pred_lstm],
            "danger": danger_map[pred_lstm][0],
            "color": danger_map[pred_lstm][1]
        }
    }
