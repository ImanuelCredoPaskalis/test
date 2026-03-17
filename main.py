import streamlit as st
import cv2
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Wajah & Mata", page_icon="📷")

st.title("📷 Deteksi Wajah dan Mata")
st.write("Ambil foto menggunakan webcam untuk mendeteksi wajah dan mata.")

# --- Fungsi Load Classifier ---
@st.cache_resource
def load_cascades():
    face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    
    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)
    
    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_cascades()

# --- Input Kamera ---
img_file_buffer = st.camera_input("Ambil foto")

if img_file_buffer is not None:
    # Konversi buffer ke OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Proses Deteksi
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi Wajah
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Gambar kotak wajah (Biru)
        cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = cv2_img[y:y+h, x:x+w]
        
        # Deteksi Mata
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Gambar kotak mata (Hijau)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Tampilkan Hasil (Pastikan baris ini tertutup kutipnya)
    st.image(cv2_img, channels="BGR", caption="Hasil Deteksi")
    
    if len(faces) > 0:
        st.success(f"Terdeteksi {len(faces)} wajah!")
    else:
        st.warning("Wajah tidak terdeteksi.")

st.divider()
st.caption("Aplikasi Deteksi OpenCV")
