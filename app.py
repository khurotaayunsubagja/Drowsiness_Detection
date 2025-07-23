# app.py
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import time
import threading
from playsound import playsound

# ⚠️ WAJIB: Harus di paling atas sebelum perintah Streamlit lainnya
st.set_page_config(page_title="Deteksi Drowsy Realtime", layout="centered")

# Fungsi untuk mainkan suara alarm
def mainkan_suara_drowsy():
    threading.Thread(target=lambda: playsound("C:/Users/lenovvo/alarm-restricted-access-355278.mp3")).start()

# Load model YOLOv5 custom
@st.cache_resource
def load_model():
    model = torch.hub.load(
        'ultralytics/yolov5', 'custom',
        path=r"C:\Users\lenovvo\Downloads\your_repo\best.pt",
        force_reload=False,
        device='cpu'
    )
    return model

model = load_model()

# UI
st.title("🚨 Deteksi Drowsy (Mengantuk) Realtime")
st.markdown("Klik tombol **Start Deteksi** untuk mengaktifkan kamera dan memulai deteksi mata mengantuk secara langsung.")

# Simpan status di session_state
if 'deteksi_aktif' not in st.session_state:
    st.session_state.deteksi_aktif = False

# Tombol kontrol kamera
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Deteksi"):
        st.session_state.deteksi_aktif = True
with col2:
    if st.button("⏹️ Stop"):
        st.session_state.deteksi_aktif = False

# Tampilkan frame webcam
FRAME_WINDOW = st.image([])

# Parameter deteksi
cooldown = 3
last_sound_time = 0
frame_drowsy_tidak_terdeteksi = 0
drowsy_active = False
CONFIDENCE_THRESHOLD = 0.3

# Jalankan deteksi jika tombol Start ditekan
if st.session_state.deteksi_aktif:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    st.success("📸 Kamera aktif. Deteksi berjalan...")

    while st.session_state.deteksi_aktif:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Tidak bisa membaca frame dari kamera.")
            break

        results = model(frame)
        df = results.pandas().xyxy[0]
        frame_bgr = np.squeeze(results.ims[0])

        current_time = time.time()
        drowsy_detected = False

        for i, row in df.iterrows():
            if row['confidence'] > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = f"{row['name']} {row['confidence']:.2f}"

                # Gambar bounding box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Tambahkan label
                cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                # Deteksi mengantuk
                if row['name'] == 'drowsy':
                    drowsy_detected = True

        if drowsy_detected:
            frame_drowsy_tidak_terdeteksi = 0
            if not drowsy_active or (current_time - last_sound_time > cooldown):
                mainkan_suara_drowsy()
                last_sound_time = current_time
                drowsy_active = True
        else:
            frame_drowsy_tidak_terdeteksi += 1
            if frame_drowsy_tidak_terdeteksi >= 10:
                drowsy_active = False

        # Konversi ke RGB dan tampilkan
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    cap.release()
else:
    st.info("Klik 'Start Deteksi' untuk memulai.")
