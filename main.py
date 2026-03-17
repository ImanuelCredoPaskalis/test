import streamlit as st
import cv2
import numpy as np

# Konfigurasi Halaman (Agar muncul judul di tab browser)
st.set_page_config(page_title="Deteksi Wajah & Mata", page_icon="📷")

st.title("📷 Deteksi Wajah dan Mata")
st.write("Ambil foto menggunakan webcam untuk mendeteksi wajah (kotak biru) dan mata (kotak hijau).")

# --- Fungsi Load Classifier (Menggunakan Cache agar Cepat) ---
@st.cache_resource
def load_cascades():
    # Mengambil path XML bawaan dari library OpenCV yang terinstal
    face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    
    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)
    
    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_cascades()

# --- Sidebar untuk Pengaturan ---
st.sidebar.header("🔧 Pengaturan Deteksi")
scale_factor = st.sidebar.slider("Sensitivitas (Scale Factor)", 1.1, 1.5, 1.3, 0.1)
min_neighbors = st.sidebar.slider("Akurasi (Min Neighbors)", 3, 10, 5)

# --- Area Input Kamera ---
img_file_buffer = st.camera_input("Klik tombol di bawah untuk ambil foto")

if img_file_buffer is not None:
    # 1. Konversi foto dari buffer ke format yang dimengerti OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 2. Proses Deteksi
    with st.spinner("Sedang memproses gambar..."):
        # Ubah ke Grayscale (hitam putih) karena deteksi Haar Cascade butuh itu
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)

        for (x, y, w, h) in faces:
            # Gambar kotak wajah (Warna Biru: BGR = 255, 0, 0)
            cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
            # Tentukan area wajah saja untuk mencari mata (ROI)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = cv2_img[y:y+h, x:x+w]
            
            # Deteksi Mata di dalam area wajah
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # Gambar kotak mata (Warna Hijau: BGR = 0, 255, 0)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # 3. Tampilkan Hasil Akhir
    st.image(cv2_img, channels="
