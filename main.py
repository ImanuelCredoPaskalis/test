import streamlit as st
import cv2
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Wajah & Mata", page_icon="📷")

st.title("📷 Deteksi Wajah dan Mata")
st.write("Aplikasi ini menggunakan OpenCV untuk mendeteksi wajah dan mata secara real-time dari foto webcam.")

# --- Fungsi Load Classifier ---
@st.cache_resource # Gunakan cache agar tidak loading ulang setiap kali script berjalan
def load_cascades():
    # Mengambil path bawaan OpenCV
    face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    
    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)
    
    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_cascades()

# --- Sidebar untuk Pengaturan ---
st.sidebar.header("Pengaturan Deteksi")
scale_factor = st.sidebar.slider("Scale Factor", 1.1, 1.5, 1.3, 0.1)
min_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5)

# --- Input Kamera ---
img_file_buffer = st.camera_input("Ambil foto untuk mulai mendeteksi")

if img_file_buffer is not None:
    # Konversi buffer ke OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Proses Deteksi
    with st.spinner("Sedang mendeteksi..."):
        # Convert ke grayscale
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)

        for (x, y, w, h) in faces:
            # Gambar kotak wajah (Warna Biru)
            cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
            # Region of Interest (ROI) untuk mata di dalam area wajah
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = cv2_img[y:y+h, x:x+w]
            
            # Deteksi Mata
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # Gambar kotak mata (Warna Hijau)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Tampilkan Hasil
    st.image(cv2_img, channels="BGR", caption="Hasil Deteksi")
    
    if len(faces) > 0:
        st.success(f"Terdeteksi {len(faces)} wajah!")
    else:
        st.warning("Wajah tidak terdeteksi. Coba pencahayaan yang lebih baik.")
else:
    st.info("Silakan klik tombol 'Take Photo' di atas.")

---
st.caption("Dibuat dengan Streamlit dan OpenCV")
