import streamlit as st

# Konfigurasi Halaman
st.set_page_config(page_title="Kalkulator Streamlit", page_icon="🧮")

st.title("🧮 Kalkulator Sederhana")
st.write("Masukkan angka dan pilih operasi matematika yang diinginkan.")

# Input Angka
col1, col2 = st.columns(2)
with col1:
    angka1 = st.number_input("Masukkan Angka Pertama", value=0.0)
with col2:
    angka2 = st.number_input("Masukkan Angka Kedua", value=0.0)

st.divider()

# Pilihan Operasi
operasi = st.selectbox("Pilih Operasi:", ("Tambah (+)", "Kurang (-)", "Kali (x)", "Bagi (/)"))

hasil = 0

# Logika Perhitungan
if st.button("Hitung Hasil"):
    if operasi == "Tambah (+)":
        hasil = angka1 + angka2
        st.success(f"Hasil dari {angka1} + {angka2} = {hasil}")
        
    elif operasi == "Kurang (-)":
        hasil = angka1 - angka2
        st.success(f"Hasil dari {angka1} - {angka2} = {hasil}")
        
    elif operasi == "Kali (x)":
        hasil = angka1 * angka2
        st.success(f"Hasil dari {angka1} x {angka2} = {hasil}")
        
    elif operasi == "Bagi (/)":
        if angka2 != 0:
            hasil = angka1 / angka2
            st.success(f"Hasil dari {angka1} / {angka2} = {hasil}")
        else:
            st.error("Error: Tidak bisa membagi dengan angka nol!")

st.divider()
st.caption("Dibuat dengan Streamlit")
