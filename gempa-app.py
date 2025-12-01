import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime, timedelta
import pytz
import numpy as np
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi & Prediksi Kedalaman Gempa",
    page_icon="🌍",
    layout="wide"
)

# Load model klasifikasi kedalaman
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_depth_class.pkl")

model = load_model()

# Fungsi fetch data
def fetch_usgs_indonesia_earthquakes():
    try:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            'format': 'geojson',
            'starttime': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'endtime': datetime.now().strftime('%Y-%m-%d'),
            'minlatitude': -11,
            'maxlatitude': 6,
            'minlongitude': 95,
            'maxlongitude': 141,
            'minmagnitude': 2.5,
            'limit': 100
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        earthquakes = []
        
        for feature in data['features']:
            try:
                properties = feature['properties']
                geometry = feature['geometry']
                
                magnitude = properties.get('mag', 0)
                place = properties.get('place', 'Unknown location')
                time_ms = properties.get('time', 0)
                depth = geometry['coordinates'][2] if len(geometry['coordinates']) > 2 else 0
                
                utc_time = pd.to_datetime(time_ms, unit='ms')
                local_time = utc_time.tz_localize('UTC').tz_convert(pytz.timezone('Asia/Jakarta'))
                
                earthquakes.append({
                    "tanggal": local_time.strftime("%d-%b-%Y"),
                    "jam": local_time.strftime("%H:%M:%S"),
                    "lintang": geometry['coordinates'][1],
                    "bujur": geometry['coordinates'][0],
                    "magnitudo": magnitude,
                    "kedalaman": int(depth),
                    "wilayah": place,
                    "potensi_tsunami": "Tidak berpotensi tsunami" if magnitude < 7.0 else "Berpotensi tsunami",
                    "waktu_kejadian": local_time
                })
                
            except:
                continue
        
        return pd.DataFrame(earthquakes)
        
    except:
        st.error("Gagal mengambil data USGS.")
        return pd.DataFrame()


# HEADER
st.title("🌍 Deteksi Gempa + Prediksi Kedalaman Gempa")
st.markdown("Mengambil data gempa real-time dari USGS dan memprediksi kategori kedalaman gempa menggunakan model Machine Learning.")

# Ambil data gempa
with st.spinner("📡 Mengambil data gempa..."):
    earthquake_data = fetch_usgs_indonesia_earthquakes()

# Informasi sumber
if not earthquake_data.empty:
    st.info("📊 **Sumber Data:** USGS — Gempa 7 hari terakhir wilayah Indonesia")

# Statistik ringkas
if not earthquake_data.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Gempa", len(earthquake_data))
    col2.metric("Magnitudo Tertinggi", f"{earthquake_data['magnitudo'].max():.1f}")
    col3.metric("Kedalaman Rata-rata", f"{earthquake_data['kedalaman'].mean():.0f} km")
    col4.metric("Gempa M ≥ 4.0", len(earthquake_data[earthquake_data['magnitudo'] >= 4.0]))

# Filter berdasarkan magnitudo
if not earthquake_data.empty:
    max_magnitude = float(earthquake_data['magnitudo'].max())
    min_magnitude = st.slider(
        "Magnitudo Minimum", 
        min_value=0.0, 
        max_value=max_magnitude, 
        value=2.5, 
        step=0.1
    )
    filtered_data = earthquake_data[earthquake_data["magnitudo"] >= min_magnitude]
else:
    filtered_data = earthquake_data

# PETA
if not filtered_data.empty:
    st.subheader("🗺️ Peta Gempa Bumi")
    
    fig = px.scatter_mapbox(
        filtered_data,
        lat="lintang",
        lon="bujur",
        size="magnitudo",
        color="magnitudo",
        hover_name="wilayah",
        hover_data=["tanggal", "jam", "magnitudo", "kedalaman"],
        zoom=4,
        center={"lat": -2.5, "lon": 118},
        height=600,
        color_continuous_scale="Reds"
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================
# 🔮 PREDIKSI KEDALAMAN GEMPA (FITUR TRAINING KAMU)
# ===========================================================
st.markdown("---")
st.header("🔮 Prediksi Kedalaman Gempa (Machine Learning)")

st.write("Masukkan parameter gempa untuk memprediksi kategori kedalaman:")

colA, colB, colC = st.columns(3)

magnitudo_input = colA.number_input("Magnitudo", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
lintang_input = colB.number_input("Lintang", min_value=-11.0, max_value=6.0, value=-2.5, step=0.01)
bujur_input = colC.number_input("Bujur", min_value=95.0, max_value=141.0, value=118.0, step=0.01)

if st.button("Prediksi Kedalaman"):
    fitur = np.array([[magnitudo_input, lintang_input, bujur_input]])
    pred = model.predict(fitur)[0]
    
    mapping = {
        0: "Shallow (<70 km)",
        1: "Intermediate (70–300 km)",
        2: "Deep (>300 km)"
    }
    
    hasil = mapping.get(pred, "Tidak diketahui")
    st.success(f"📌 **Hasil Prediksi:** {hasil}")


# TABEL
st.subheader("📋 Data Gempa Terkini")

if not filtered_data.empty:
    data_show = filtered_data.sort_values("waktu_kejadian", ascending=False)
    st.dataframe(
        data_show[['tanggal', 'jam', 'wilayah', 'magnitudo', 'kedalaman', 'potensi_tsunami']],
        use_container_width=True,
        height=400
    )

    csv = data_show.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        "gempa_indonesia.csv",
        "text/csv"
    )

# Sidebar
st.sidebar.info("Aplikasi ini memantau gempa real-time dan memprediksi kategori kedalaman gempa (Shallow, Intermediate, Deep).")
