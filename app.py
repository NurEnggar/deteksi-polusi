import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 200
    data = {
        "Temperature": np.random.normal(30, 5, n).round(1),
        "Kelembapan": np.random.normal(70, 10, n).round(1),
        "Kecepatan Angin": np.random.normal(10, 3, n).round(1),
        "CO": np.random.uniform(0.5, 5.0, n).round(2),
        "NO2": np.random.uniform(10, 80, n).round(1),
    }

    df = pd.DataFrame(data)

    # PM2.5 dengan variasi lebih luas dan realistis
    df["PM2.5"] = (
        3.0 * df["CO"]
        + 0.8 * df["NO2"]
        - 0.5 * df["Kecepatan Angin"]
        + 0.3 * df["Temperature"]
        - 0.1 * df["Kelembapan"]
        + 25  # baseline
        + np.random.normal(0, 10, n)
    ).round(1).clip(lower=0)

    # Kategori AQI berdasarkan PM2.5
    def categorize_pm25(x):
        if x <= 30:
            return "Bagus"
        elif x <= 60:
            return "Moderate"
        elif x <= 100:
            return "Tidak sehat"
        else:
            return "Berbahaya"

    df["AQI"] = df["PM2.5"].apply(categorize_pm25)
    return df

# Load data dan latih model
df = load_data()
X = df.drop(columns=["PM2.5", "AQI"])
y = df["PM2.5"]

model = RandomForestRegressor()
model.fit(X, y)

# UI Streamlit
st.title("Prediksi Kualitas Udara Berdasarkan Lingkungan")
# ------------------ UI Streamlit ------------------
st.title("🧪 Prediksi Kualitas Udara Berdasarkan Parameter Lingkungan")

st.markdown("""
Aplikasi ini digunakan untuk **memprediksi kadar PM2.5** (partikel polutan udara berukuran <2.5 µm) dan mengklasifikasikan kualitas udara (**Air Quality Index/AQI**) berdasarkan parameter lingkungan seperti:

- Suhu udara (Temperature)
- Kelembapan
- Kecepatan angin
- Konsentrasi CO (karbon monoksida)
- Konsentrasi NO₂ (nitrogen dioksida)

---

### 🎯 Tujuan:
- Memberikan estimasi kadar PM2.5 berdasarkan data lingkungan.
- Menyediakan **klasifikasi kualitas udara** agar pengguna dapat mengetahui kondisi udara saat ini atau prakiraan ke depan.

---

### 🧠 Metode:
Model menggunakan **Random Forest Regressor**, yaitu algoritma **ensemble learning** yang membangun banyak pohon keputusan (decision trees) dan mengambil rata-rata hasilnya untuk prediksi numerik.

Metode ini dipilih karena:
- Akurat untuk regresi non-linear
- Tahan terhadap overfitting
- Cocok untuk data simulasi seperti ini

---
""")
st.markdown("Masukkan parameter lingkungan untuk memprediksi nilai PM2.5 dan kualitas udara:")

temp = st.slider("Temperature (°C)", 10.0, 45.0, 30.0)
humidity = st.slider("Kelembapan (%)", 10.0, 100.0, 70.0)
wind = st.slider("Kecepatan Angin (km/h)", 0.0, 30.0, 10.0)
co = st.slider("CO (ppm)", 0.1, 10.0, 1.0)
no2 = st.slider("NO2 (ppb)", 0.0, 200.0, 40.0)

input_data = pd.DataFrame([[temp, humidity, wind, co, no2]],
                          columns=["Temperature", "Kelembapan", "Kecepatan Angin", "CO", "NO2"])

predicted_pm25 = model.predict(input_data)[0]

# Konversi ke kategori AQI
if predicted_pm25 <= 30:
    aqi = "Bagus"
elif predicted_pm25 <= 60:
    aqi = "Moderate"
elif predicted_pm25 <= 100:
    aqi = "Tidak sehat"
else:
    aqi = "Berbahaya"

# Tampilkan hasil
st.subheader("Hasil Prediksi:")
st.write(f"PM2.5 diperkirakan: **{predicted_pm25:.1f} µg/m³**")
st.write(f"Kualitas udara diperkirakan: **{aqi}**")

# Tampilkan data latih jika diminta
if st.checkbox("Tampilkan Data Latih"):
    st.dataframe(df)

# Tombol unduh dataset
if st.button("Unduh Dataset"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download data sebagai CSV",
        data=csv,
        file_name="data_kualitas_udara.csv",
        mime="text/csv"
    )
