import time
import threading
from pathlib import Path

import av
import cv2
import torch
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
    RTCConfiguration,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="😴",
    layout="wide"
)


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"

ALARM_PATH = (
    BASE_DIR /
    "alarm-restricted-access-355278.mp3"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error(
        "Model best.pt tidak ditemukan."
    )
    st.stop()


model = load_model()


# ============================================================
# HELPER
# ============================================================

DROWSY_KEYWORDS = [
    "drowsy",
    "sleepy",
    "sleep",
    "closed",
    "close",
    "ngantuk",
]


def get_class_name(names, class_id):

    if isinstance(names, dict):
        return str(
            names.get(class_id, class_id)
        )

    return str(names[class_id])


def is_drowsy_label(label):

    label = str(label).lower()

    return any(
        keyword in label
        for keyword in DROWSY_KEYWORDS
    )


def detect_drowsiness(result):
    """
    Mendukung YOLO Detection maupun
    YOLO Classification.
    """

    best_label = "Normal"
    best_confidence = 0.0
    drowsy = False

    names = result.names

    # ========================================================
    # OBJECT DETECTION
    # ========================================================

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            label = get_class_name(
                names,
                class_id
            )

            # Simpan detection confidence tertinggi
            if confidence > best_confidence:

                best_confidence = confidence
                best_label = label

            # Cek apakah ada kelas ngantuk
            if is_drowsy_label(label):
                drowsy = True

    # ========================================================
    # IMAGE CLASSIFICATION
    # ========================================================

    elif result.probs is not None:

        class_id = int(
            result.probs.top1
        )

        confidence = float(
            result.probs.top1conf.item()
        )

        label = get_class_name(
            names,
            class_id
        )

        best_label = label
        best_confidence = confidence

        if is_drowsy_label(label):
            drowsy = True

    return (
        drowsy,
        best_label,
        best_confidence
    )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class DrowsinessProcessor(
    VideoProcessorBase
):

    def __init__(
        self,
        confidence_threshold,
        drowsy_duration
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

        self.drowsy_duration = (
            drowsy_duration
        )

        self.drowsy_start = None

        self.alert = False

        self.label = "Normal"
        self.confidence = 0.0
        self.duration = 0.0

        self.lock = threading.Lock()


    def recv(self, frame):

        # Convert WebRTC frame ke OpenCV
        image = frame.to_ndarray(
            format="bgr24"
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        results = model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False
        )

        result = results[0]


        # ====================================================
        # CHECK RESULT
        # ====================================================

        (
            drowsy,
            label,
            confidence
        ) = detect_drowsiness(result)


        # ====================================================
        # DROWSINESS TIMER
        # ====================================================

        current_time = time.monotonic()

        if drowsy:

            if self.drowsy_start is None:

                self.drowsy_start = (
                    current_time
                )

            duration = (
                current_time
                - self.drowsy_start
            )

            alert = (
                duration
                >= self.drowsy_duration
            )

        else:

            self.drowsy_start = None

            duration = 0.0
            alert = False


        # ====================================================
        # SAVE STATUS
        # ====================================================

        with self.lock:

            self.alert = alert

            self.label = label

            self.confidence = confidence

            self.duration = duration


        # ====================================================
        # DRAW YOLO RESULT
        # ====================================================

        annotated_frame = result.plot()


        # ====================================================
        # STATUS TEXT
        # ====================================================

        if alert:

            status_text = (
                "WARNING: DROWSINESS DETECTED!"
            )

            text_color = (0, 0, 255)

        elif drowsy:

            status_text = (
                f"Drowsy: {duration:.1f}s"
            )

            text_color = (0, 165, 255)

        else:

            status_text = "Status: Awake"

            text_color = (0, 255, 0)


        cv2.putText(
            annotated_frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2,
            cv2.LINE_AA
        )


        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


    def get_status(self):

        with self.lock:

            return {
                "alert": self.alert,
                "label": self.label,
                "confidence": self.confidence,
                "duration": self.duration,
            }


# ============================================================
# HEADER
# ============================================================

st.title(
    "😴 Drowsiness Detection System"
)

st.write(
    """
    Sistem ini menggunakan computer vision untuk
    mendeteksi indikasi kantuk secara real-time
    melalui webcam.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05
    )

    drowsy_duration = st.slider(
        "Drowsiness Duration (seconds)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )

    st.divider()

    st.subheader(
        "Model Classes"
    )

    st.write(model.names)


# ============================================================
# INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    status_metric = st.empty()

with col2:
    confidence_metric = st.empty()

with col3:
    duration_metric = st.empty()


status_metric.metric(
    "Status",
    "Waiting"
)

confidence_metric.metric(
    "Confidence",
    "-"
)

duration_metric.metric(
    "Drowsy Duration",
    "0.0 s"
)


# ============================================================
# WEBRTC CONFIG
# ============================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# ============================================================
# CAMERA
# ============================================================

st.subheader(
    "📷 Live Camera"
)

webrtc_ctx = webrtc_streamer(

    key="drowsiness-detection",

    mode=WebRtcMode.SENDRECV,

    rtc_configuration=RTC_CONFIGURATION,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    video_processor_factory=lambda:
        DrowsinessProcessor(
            confidence_threshold,
            drowsy_duration
        ),

    async_processing=True
)


# ============================================================
# ALERT AREA
# ============================================================

alert_container = st.empty()

alarm_container = st.empty()


# ============================================================
# REAL-TIME STATUS
# ============================================================

if webrtc_ctx.state.playing:

    alarm_playing = False

    while webrtc_ctx.state.playing:

        processor = (
            webrtc_ctx.video_processor
        )

        if processor is not None:

            data = processor.get_status()

            label = data["label"]

            confidence = (
                data["confidence"]
            )

            duration = data["duration"]

            alert = data["alert"]


            # ================================================
            # UPDATE METRICS
            # ================================================

            if alert:

                status_metric.metric(
                    "Status",
                    "🚨 DROWSY"
                )

            elif duration > 0:

                status_metric.metric(
                    "Status",
                    "⚠️ Drowsiness detected"
                )

            else:

                status_metric.metric(
                    "Status",
                    "✅ Awake"
                )


            confidence_metric.metric(
                "Confidence",
                f"{confidence:.1%}"
            )

            duration_metric.metric(
                "Drowsy Duration",
                f"{duration:.1f} s"
            )


            # ================================================
            # ALARM
            # ================================================

            if alert:

                alert_container.error(
                    "🚨 WARNING! "
                    "Drowsiness detected. "
                    "Please stay alert!"
                )

                if (
                    not alarm_playing
                    and ALARM_PATH.exists()
                ):

                    alarm_bytes = (
                        ALARM_PATH.read_bytes()
                    )

                    alarm_container.audio(
                        alarm_bytes,
                        format="audio/mp3",
                        autoplay=True
                    )

                    alarm_playing = True

            else:

                alert_container.empty()

                if alarm_playing:

                    alarm_container.empty()

                    alarm_playing = False


        time.sleep(0.25)
