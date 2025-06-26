import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Prediksi Kualitas Udara (AQI)")

# Deskripsi aplikasi
st.markdown("""
Aplikasi **Prediksi Kualitas Udara (AQI)** ini dirancang untuk memprediksi tingkat kualitas udara berdasarkan beberapa parameter lingkungan. 
Pengguna dapat mengatur nilai input melalui slider yang tersedia dan mendapatkan hasil prediksi secara langsung.

### Fitur Utama:
- **Input Parameter Lingkungan:**
    - Temperature (°C)
    - Kelembapan (%)
    - Kecepatan Angin (km/h)
    - CO (ppm)
    - NO2 (ppb)
    - PM2.5 (µg/m³)

- **Prediksi Kualitas Udara:**
    Berdasarkan nilai yang diinputkan, model machine learning akan memprediksi kualitas udara ke dalam beberapa kategori seperti:
    - Baik (Good)
    - Sedang (Moderate)
    - Tidak Sehat (Unhealthy), dan lainnya.

- **Opsi Tambahan:**
    - Menampilkan data latih (jika dicentang)

### Metode yang Digunakan:
Model prediksi dalam aplikasi ini menggunakan algoritma **Random Forest Classifier**, 
yaitu salah satu metode machine learning berbasis ensemble yang menggabungkan banyak pohon keputusan (*decision trees*). 
Metode ini dipilih karena kemampuannya yang baik dalam menangani data non-linear, robust terhadap outlier, 
dan memberikan hasil prediksi yang akurat dalam berbagai jenis data lingkungan.

### Tujuan Aplikasi:
Aplikasi ini bertujuan untuk membantu pengguna memahami pengaruh parameter lingkungan terhadap kualitas udara. 
Cocok digunakan untuk edukasi, riset, maupun pemantauan kondisi lingkungan sehari-hari.

---
""")
# Dataset
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 100
    data = {
        "Temperature": np.random.normal(30, 5, n).round(1),
        "Kelembapan": np.random.normal(70, 10, n).round(1),
        "Kecepatan Angin": np.random.normal(10, 3, n).round(1),
        "CO": np.random.uniform(0.5, 5.0, n).round(2),
        "NO2": np.random.uniform(10, 80, n).round(1),
        "PM2.5": np.random.uniform(10, 150, n).round(1),
    }
    df = pd.DataFrame(data)
    df["AQI"] = df["PM2.5"].apply(lambda x: "Bagus" if x <= 30 else "Moderate" if x <= 60 else "Tidak sehat" if x <= 100 else "Berbahaya")
    return df

df = load_data()

# Train Model
X = df.drop(columns=["AQI"])
y = df["AQI"]
model = RandomForestClassifier()
model.fit(X, y)

# UI
st.markdown("Masukkan parameter lingkungan untuk memprediksi kualitas udara:")

temp = st.slider("Temperature (°C)", 10.0, 45.0, 30.0)
humidity = st.slider("Kelembapan (%)", 10.0, 100.0, 70.0)
wind = st.slider("Kecepatan Angin (km/h)", 0.0, 30.0, 10.0)
co = st.slider("CO (ppm)", 0.1, 10.0, 1.0)
no2 = st.slider("NO2 (ppb)", 0.0, 200.0, 40.0)
pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 50.0)

input_data = pd.DataFrame([[temp, humidity, wind, co, no2, pm25]],
                          columns=["Temperature", "Kelembapan", "Kecepatan Angin", "CO", "NO2", "PM2.5"])

prediction = model.predict(input_data)[0]

st.subheader("Hasil Prediksi:")
st.write(f"Kualitas udara diperkirakan: **{prediction}**")

if st.checkbox("Tampilkan Data Latih"):
    st.dataframe(df)
