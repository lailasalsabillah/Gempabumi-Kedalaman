# Aplikasi Streamlit: klasifikasi kedalaman gempa
# Ganti file ini pada repo untuk memperbarui UI dan kemampuan upload CSV
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Klasifikasi Kedalaman Gempa", layout="centered")

def classify_depth(depth_km):
    """
    Klasifikasi kedalaman gempa dalam kilometer:
    - Dangkal: depth < 70 km
    - Menengah: 70 <= depth <= 300 km
    - Dalam: depth > 300 km
    """
    try:
        d = float(depth_km)
    except:
        return "Tidak valid"
    if d < 70:
        return "Dangkal (<70 km)"
    elif d <= 300:
        return "Menengah (70–300 km)"
    else:
        return "Dalam (>300 km)"

st.title("Klasifikasi Kedalaman Gempa Bumi")
st.write(
    "Aplikasi sederhana untuk mengklasifikasikan kedalaman gempa (km). "
    "Mode: Manual atau Upload CSV. CSV harus mempunyai kolom 'depth' atau 'depth_km' dalam satuan kilometer."
)

st.sidebar.header("Pengaturan")
input_mode = st.sidebar.selectbox("Mode Input", ["Manual", "Upload CSV"])
show_sample = st.sidebar.checkbox("Tampilkan contoh CSV & unduh", value=False)

if show_sample:
    sample_df = pd.DataFrame({"depth_km": [5, 50, 120, 230, 400]})
    st.markdown("Contoh format CSV (kolom: depth_km):")
    st.dataframe(sample_df)
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button("Unduh contoh CSV", data=csv_bytes, file_name="sample_depth.csv", mime="text/csv")

if input_mode == "Manual":
    depth = st.slider("Pilih kedalaman gempa (km):", min_value=0.0, max_value=700.0, value=10.0, step=0.1)
    label = classify_depth(depth)
    st.subheader("Hasil Klasifikasi")
    st.markdown(f"- Kedalaman: **{depth:.1f} km**\n- Kategori: **{label}**")
    st.info("Aturan klasifikasi:\n- Dangkal: < 70 km\n- Menengah: 70–300 km\n- Dalam: > 300 km")
else:
    uploaded_file = st.file_uploader("Unggah CSV (kolom: depth atau depth_km)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Normalisasi nama kolom
            if "depth" in df.columns and "depth_km" not in df.columns:
                df["depth_km"] = df["depth"]
            if "depth_km" not in df.columns:
                st.error("Kolom tidak ditemukan. Harap sertakan kolom 'depth' atau 'depth_km' (dalam km).")
                st.stop()

            df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
            before = len(df)
            df = df.dropna(subset=["depth_km"]).reset_index(drop=True)
            after = len(df)
            if after < before:
                st.warning(f"{before-after} baris dibuang karena nilai kedalaman tidak valid.")

            df["kategori"] = df["depth_km"].apply(classify_depth)

            st.subheader("Preview Data")
            st.dataframe(df.head(200))

            st.subheader("Ringkasan Kategori")
            counts = df["kategori"].value_counts().reindex([
                "Dangkal (<70 km)",
                "Menengah (70–300 km)",
                "Dalam (>300 km)"
            ]).fillna(0).astype(int)
            st.table(counts.rename("jumlah"))

            st.subheader("Histogram Kedalaman")
            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(df["depth_km"], bins=30, color="#2b8cbe", edgecolor="black")
            ax.set_xlabel("Kedalaman (km)")
            ax.set_ylabel("Frekuensi")
            ax.set_title("Distribusi Kedalaman Gempa")
            st.pyplot(fig)

            # Tombol unduh hasil klasifikasi
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Unduh hasil klasifikasi (CSV)", data=csv_out, file_name="hasil_klasifikasi_depth.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Gagal memproses file CSV: {e}")
    else:
        st.info("Unggah file CSV untuk melihat analisis. Atau beralih ke mode Manual.")

st.sidebar.markdown("---")
st.sidebar.markdown("Pengembang: Anda — Klasifikasi berdasarkan rentang kedalaman standar (dangkal/menengah/dalam).")
