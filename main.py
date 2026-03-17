import streamlit as st
import cv2
import numpy as np
import os

st.title("Face and Eye Detection with Streamlit")

# --- Load Haar Cascade Classifiers ---
# In a Streamlit app, you might want to place these files in the same directory
# as your app.py, or provide a full path.
# For demonstration, we'll assume they are available or downloaded.
# A robust app would include logic to download them if not found.

# Create a temporary directory to store haarcascades if needed (e.g., for deployment environments)
# if not os.path.exists('haarcascades'):
#     os.makedirs('haarcascades')

# You might need to adjust paths or download these files if they are not present
# Example of downloading if needed (uncomment and modify if your environment doesn't have them):
# !wget -P haarcascades/ https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
# !wget -P haarcascades/ https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty():
    st.error(f"Error: Face cascade not loaded. Check path: {face_cascade_path}")
    st.stop()
if eye_cascade.empty():
    st.error(f"Error: Eye cascade not loaded. Check path: {eye_cascade_path}")
    st.stop()

st.write("--- Take a photo from your webcam ---")

# Streamlit component for camera input
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as bytes:
    bytes_data = img_file_buffer.getvalue()

    # Convert to OpenCV format
    np_array = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    st.write("Photo captured. Performing face and eye detection...")

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw face rectangle
        roi_gray = gray[y:y+h, x:x+w] # Region of Interest for eyes (grayscale)
        roi_color = frame[y:y+h, x:x+w] # Region of Interest for eyes (color)

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # Draw eye rectangle

    st.write("Displaying detected faces and eyes:")
    # Display the resulting image in Streamlit
    st.image(frame, channels="BGR", caption="Detected Faces and Eyes")
    st.success("Face and eye detection complete!")
else:
    st.info("Waiting for you to take a picture...")
