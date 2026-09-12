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
    layout="wide",
)


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"

ALARM_PATH = (
    BASE_DIR
    / "alarm-restricted-access-355278.mp3"
)


# ============================================================
# CHECK FILE
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "❌ File model `best.pt` tidak ditemukan."
    )
    st.stop()


# ============================================================
# LOAD YOLOv5 MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Menggunakan YOLOv5 v7.0 karena best.pt
    # berasal dari YOLOv5
    model = torch.hub.load(
        "ultralytics/yolov5:v7.0",
        "custom",
        path=str(MODEL_PATH),
        force_reload=False,
        trust_repo=True,
        device="cpu",
    )

    model.eval()

    return model


try:
    model = load_model()

except Exception as e:

    st.error(
        "❌ Model YOLOv5 gagal dimuat."
    )

    st.exception(e)

    st.stop()


# ============================================================
# MODEL CLASS HELPER
# ============================================================

def get_model_classes():

    names = model.names

    if isinstance(names, dict):

        return [
            str(names[key])
            for key in sorted(names.keys())
        ]

    return [
        str(name)
        for name in names
    ]


def get_class_name(class_id):

    names = model.names

    class_id = int(class_id)

    if isinstance(names, dict):

        return str(
            names.get(
                class_id,
                class_id
            )
        )

    return str(
        names[class_id]
    )


CLASS_NAMES = get_model_classes()


# ============================================================
# AUTO FIND DROWSY CLASS
# ============================================================

DROWSY_KEYWORDS = [
    "drowsy",
    "sleepy",
    "sleep",
    "closed",
    "close",
    "ngantuk",
    "tired",
]


def find_default_drowsy_class():

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        lower_name = (
            class_name
            .lower()
            .strip()
        )

        for keyword in DROWSY_KEYWORDS:

            if keyword in lower_name:

                return index

    return 0


DEFAULT_DROWSY_INDEX = (
    find_default_drowsy_class()
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "😴 Drowsiness Detection System"
)

st.caption(
    "Real-time drowsiness detection using "
    "YOLOv5 and webcam."
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Detection Settings"
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )

    drowsy_duration = st.slider(
        "Drowsiness Duration",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help=(
            "Alarm akan aktif apabila kondisi "
            "mengantuk terdeteksi selama durasi "
            "ini."
        ),
    )

    st.divider()

    st.subheader(
        "😴 Drowsy Class"
    )

    drowsy_class = st.selectbox(
        "Pilih kelas yang menunjukkan kondisi mengantuk",
        options=CLASS_NAMES,
        index=DEFAULT_DROWSY_INDEX,
    )

    st.divider()

    st.subheader(
        "📦 Model Classes"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        st.write(
            f"{i}: {name}"
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
        drowsy_class,
        drowsy_duration,
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

        self.drowsy_class = (
            str(drowsy_class)
        )

        self.drowsy_duration = (
            drowsy_duration
        )

        self.drowsy_start = None

        self.alert = False

        self.current_label = (
            "No Detection"
        )

        self.current_confidence = 0.0

        self.current_duration = 0.0

        self.lock = threading.Lock()


    def recv(
        self,
        frame
    ):

        # ====================================================
        # GET CAMERA FRAME
        # ====================================================

        image = frame.to_ndarray(
            format="bgr24"
        )

        # YOLO menerima RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # YOLOv5 PREDICTION
        # ====================================================

        with torch.no_grad():

            results = model(
                rgb_image,
                size=640
            )


        # Format:
        # x1, y1, x2, y2, confidence, class
        detections = (
            results
            .xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )


        # ====================================================
        # INITIAL STATUS
        # ====================================================

        is_drowsy = False

        best_label = "No Detection"

        best_confidence = 0.0


        # ====================================================
        # READ DETECTIONS
        # ====================================================

        for detection in detections:

            x1 = int(
                detection[0]
            )

            y1 = int(
                detection[1]
            )

            x2 = int(
                detection[2]
            )

            y2 = int(
                detection[3]
            )

            confidence = float(
                detection[4]
            )

            class_id = int(
                detection[5]
            )


            # Skip confidence rendah
            if (
                confidence
                < self.confidence_threshold
            ):
                continue


            label = get_class_name(
                class_id
            )


            # =================================================
            # SAVE BEST DETECTION
            # =================================================

            if (
                confidence
                > best_confidence
            ):

                best_confidence = (
                    confidence
                )

                best_label = label


            # =================================================
            # CHECK DROWSINESS
            # =================================================

            drowsy_detection = (
                label.lower().strip()
                ==
                self.drowsy_class
                .lower()
                .strip()
            )

            if drowsy_detection:

                is_drowsy = True

                box_color = (
                    0,
                    0,
                    255
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )


            label_text = (
                f"{label} "
                f"{confidence:.0%}"
            )


            cv2.putText(
                image,
                label_text,
                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
                cv2.LINE_AA,
            )


        # ====================================================
        # DROWSINESS TIMER
        # ====================================================

        current_time = (
            time.monotonic()
        )


        if is_drowsy:

            if (
                self.drowsy_start
                is None
            ):

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

            self.current_label = (
                best_label
            )

            self.current_confidence = (
                best_confidence
            )

            self.current_duration = (
                duration
            )


        # ====================================================
        # STATUS OVERLAY
        # ====================================================

        if alert:

            status_text = (
                "WARNING: DROWSINESS DETECTED!"
            )

            status_color = (
                0,
                0,
                255
            )

        elif is_drowsy:

            status_text = (
                "Drowsiness detected: "
                f"{duration:.1f}s"
            )

            status_color = (
                0,
                165,
                255
            )

        else:

            status_text = (
                "Status: Awake"
            )

            status_color = (
                0,
                255,
                0
            )


        # Background untuk text
        cv2.rectangle(
            image,
            (10, 10),
            (620, 60),
            (0, 0, 0),
            -1,
        )


        cv2.putText(
            image,
            status_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )


        # ====================================================
        # RETURN FRAME
        # ====================================================

        return (
            av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )
        )


    def get_status(self):

        with self.lock:

            return {
                "alert":
                    self.alert,

                "label":
                    self.current_label,

                "confidence":
                    self.current_confidence,

                "duration":
                    self.current_duration,
            }


# ============================================================
# STATUS CARDS
# ============================================================

col1, col2, col3 = st.columns(
    3
)

with col1:

    status_placeholder = (
        st.empty()
    )

with col2:

    confidence_placeholder = (
        st.empty()
    )

with col3:

    duration_placeholder = (
        st.empty()
    )


status_placeholder.metric(
    "Status",
    "Waiting"
)

confidence_placeholder.metric(
    "Confidence",
    "-"
)

duration_placeholder.metric(
    "Drowsy Duration",
    "0.0 s"
)


# ============================================================
# WEBRTC CONFIG
# ============================================================

RTC_CONFIGURATION = (
    RTCConfiguration(
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
)


# ============================================================
# CAMERA
# ============================================================

st.subheader(
    "📷 Live Camera"
)

st.info(
    "Klik START lalu izinkan browser "
    "mengakses kamera."
)


webrtc_ctx = webrtc_streamer(

    key="drowsiness-camera",

    mode=WebRtcMode.SENDRECV,

    rtc_configuration=(
        RTC_CONFIGURATION
    ),

    media_stream_constraints={
        "video": True,
        "audio": False,
    },

    video_processor_factory=lambda:
        DrowsinessProcessor(
            confidence_threshold=
                confidence_threshold,

            drowsy_class=
                drowsy_class,

            drowsy_duration=
                drowsy_duration,
        ),

    async_processing=True,
)


# ============================================================
# ALERT AREA
# ============================================================

alert_placeholder = st.empty()

alarm_placeholder = st.empty()


# ============================================================
# REAL-TIME INFORMATION
# ============================================================

if webrtc_ctx.state.playing:

    alarm_is_playing = False


    while (
        webrtc_ctx.state.playing
    ):

        processor = (
            webrtc_ctx.video_processor
        )


        if processor is not None:

            status = (
                processor.get_status()
            )


            alert = status[
                "alert"
            ]

            label = status[
                "label"
            ]

            confidence = status[
                "confidence"
            ]

            duration = status[
                "duration"
            ]


            # ================================================
            # STATUS
            # ================================================

            if alert:

                status_placeholder.metric(
                    "Status",
                    "🚨 DROWSY"
                )

            elif duration > 0:

                status_placeholder.metric(
                    "Status",
                    "⚠️ Drowsy"
                )

            else:

                status_placeholder.metric(
                    "Status",
                    "✅ Awake"
                )


            # ================================================
            # CONFIDENCE
            # ================================================

            if (
                confidence > 0
            ):

                confidence_placeholder.metric(
                    "Confidence",
                    f"{confidence:.1%}",
                    help=(
                        f"Detected class: "
                        f"{label}"
                    ),
                )

            else:

                confidence_placeholder.metric(
                    "Confidence",
                    "-"
                )


            # ================================================
            # DURATION
            # ================================================

            duration_placeholder.metric(
                "Drowsy Duration",
                f"{duration:.1f} s"
            )


            # ================================================
            # ALARM
            # ================================================

            if alert:

                alert_placeholder.error(
                    "🚨 DROWSINESS DETECTED! "
                    "Please stay alert."
                )


                if (
                    not alarm_is_playing
                    and ALARM_PATH.exists()
                ):

                    alarm_bytes = (
                        ALARM_PATH
                        .read_bytes()
                    )


                    alarm_placeholder.audio(
                        alarm_bytes,
                        format="audio/mpeg",
                        autoplay=True,
                    )


                    alarm_is_playing = (
                        True
                    )

            else:

                alert_placeholder.empty()


                if alarm_is_playing:

                    alarm_placeholder.empty()

                    alarm_is_playing = (
                        False
                    )


        time.sleep(
            0.25
        )
