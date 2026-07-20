"""
Age & Gender Detection — Professional AI Dashboard
====================================================
Frontend redesign only. All detection logic lives in detector.py
and is used exactly as-is via process_image().
"""

import os
import io
import base64
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

from detector import process_image

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Age & Gender Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def load_css(file_name: str):
    """Inject a local CSS file into the Streamlit app."""
    css_path = os.path.join(ASSETS_DIR, file_name)
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def img_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def get_asset_base64(filename: str) -> str:
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR image (or already-RGB) to a PIL Image safely."""
    if cv2_image is None:
        return None
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def metric_card(icon: str, label: str, value: str, accent: str = "blue"):
    st.markdown(
        f"""
        <div class="metric-card metric-{accent}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_session_state():
    defaults = {
        "processed_image": None,
        "original_image": None,
        "results": None,
        "processing_time": None,
        "timestamp": None,
        "history_runs": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()
load_css("style.css")

logo_b64 = get_asset_base64("logo.png")
banner_b64 = get_asset_base64("banner.png")

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    if logo_b64:
        st.markdown(
            f"""<div class="sidebar-logo-wrap">
                    <img src="data:image/png;base64,{logo_b64}" class="sidebar-logo"/>
                </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="sidebar-logo-wrap"><div class="sidebar-logo-fallback">🧠</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """<h2 class="sidebar-title">Age & Gender<br>Detection AI</h2>""",
        unsafe_allow_html=True,
    )

    nav = option_menu(
        menu_title=None,
        options=["About", "Features", "Tech Stack", "Model Info", "Developer"],
        icons=["info-circle", "stars", "cpu", "diagram-3", "person-badge"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#3B82F6", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px 0",
                "border-radius": "10px",
                "color": "#334155",
            },
            "nav-link-selected": {"background-color": "#3B82F6", "color": "white"},
        },
    )

    st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
    if nav == "About":
        st.markdown(
            """
            **About the Project**

            An AI-powered computer vision system that detects human faces
            in an image and predicts **age range** and **gender** in
            real time using a deep learning pipeline built on OpenCV's DNN
            module.
            """
        )
    elif nav == "Features":
        st.markdown(
            """
            **Key Features**
            - 🎯 Real-time face detection
            - 📊 Confidence scoring per prediction
            - 📸 Upload or live camera input
            - ⬇️ Downloadable processed results
            """
        )
    elif nav == "Tech Stack":
        st.markdown(
            """
            **Technology Stack**
            - 🐍 Python
            - 👁️ OpenCV (DNN Module)
            - 🎈 Streamlit
            - 📊 Plotly
            - 🐼 Pandas
            - 🖼️ Pillow
            """
        )
    elif nav == "Model Info":
        st.markdown(
            """
            **Model Information**
            - Face Detector: OpenCV DNN (Caffe/TensorFlow)
            - Age Classifier: Pretrained CNN
            - Gender Classifier: Pretrained CNN
            - Input: RGB image, auto-resized
            - Output: Bounding boxes + class confidences
            """
        )
    elif nav == "Developer":
        st.markdown(
            """
            **Developer**

            Built as an academic / portfolio project demonstrating
            applied computer vision and full-stack ML deployment.
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-footer">
            <span class="version-badge">v1.0.0</span>
            <p>© 2026 Age & Gender Detection AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# HERO BANNER
# ----------------------------------------------------------------------
banner_style = ""
if banner_b64:
    banner_style = f"background-image: linear-gradient(120deg, rgba(30,64,175,0.88), rgba(59,130,246,0.82)), url(data:image/png;base64,{banner_b64});"

st.markdown(
    f"""
    <div class="hero-banner" style="{banner_style}">
        <div class="hero-content">
            {"<img src='data:image/png;base64," + logo_b64 + "' class='hero-logo'/>" if logo_b64 else "<div class='hero-logo-fallback'>🧠</div>"}
            <h1 class="hero-title">Age &amp; Gender Detection AI</h1>
            <p class="hero-subtitle">
                A premium computer vision dashboard powered by deep learning —
                upload a photo and instantly detect faces, age range, and gender
                with confidence scoring.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">⚡ Real-time Inference</span>
                <span class="hero-badge">🎯 High Accuracy</span>
                <span class="hero-badge">📊 Interactive Analytics</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# UPLOAD SECTION
# ----------------------------------------------------------------------
st.markdown("<div class='section-title'>📤 Provide an Image</div>", unsafe_allow_html=True)
st.markdown(
    "<p class='section-subtitle'>Upload a photo from your device or capture one live with your camera.</p>",
    unsafe_allow_html=True,
)

st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
input_mode = st.radio(
    "Input source",
    options=["📁 Upload Image", "📷 Camera"],
    horizontal=True,
    label_visibility="collapsed",
)

uploaded_image = None
if input_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "Drag and drop or browse a file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
else:
    camera_file = st.camera_input("Capture", label_visibility="collapsed")
    if camera_file is not None:
        uploaded_image = Image.open(camera_file).convert("RGB")

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_image is not None:
    st.session_state["original_image"] = uploaded_image

# ----------------------------------------------------------------------
# DETECTION BUTTON
# ----------------------------------------------------------------------
st.markdown("<div class='detect-btn-wrap'>", unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([1, 1, 1])
with col_b:
    detect_clicked = st.button(
        "🚀 Run Detection",
        use_container_width=True,
        disabled=(st.session_state["original_image"] is None),
    )
st.markdown("</div>", unsafe_allow_html=True)

if detect_clicked and st.session_state["original_image"] is not None:
    with st.spinner("Analyzing image..."):
        np_image = np.array(st.session_state["original_image"])
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        processed_image, results, processing_time, timestamp = process_image(cv_image)

        st.session_state["processed_image"] = processed_image
        st.session_state["results"] = results
        st.session_state["processing_time"] = processing_time
        st.session_state["timestamp"] = timestamp
        st.session_state["history_runs"] += 1

    st.success("✅ Detection complete!")

# ----------------------------------------------------------------------
# IMAGE DISPLAY
# ----------------------------------------------------------------------
if st.session_state["original_image"] is not None:
    st.markdown("<div class='section-title'>🖼️ Image Comparison</div>", unsafe_allow_html=True)

    img_col1, img_col2 = st.columns(2, gap="large")

    with img_col1:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.markdown("<div class='image-card-header'>Original Image</div>", unsafe_allow_html=True)
        st.image(st.session_state["original_image"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with img_col2:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.markdown("<div class='image-card-header'>Processed Image</div>", unsafe_allow_html=True)
        if st.session_state["processed_image"] is not None:
            display_processed = cv2_to_pil(st.session_state["processed_image"])
            st.image(display_processed, use_container_width=True)
        else:
            st.markdown(
                "<div class='placeholder-box'>Run detection to see results here</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)



    # ----------------------------------------------------------------------
    # DOWNLOAD SECTION
    # ----------------------------------------------------------------------
    st.markdown("<div class='section-title'>⬇️ Download Results</div>", unsafe_allow_html=True)
    st.markdown("<div class='download-wrap'>", unsafe_allow_html=True)

    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
    with dl_col2:
        pil_processed = cv2_to_pil(st.session_state["processed_image"])
        buf = io.BytesIO()
        pil_processed.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Processed Image",
            data=buf.getvalue(),
            file_name=f"processed_{timestamp if timestamp else datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <h3>No Detection Yet</h3>
            <p>Upload an image and click <b>Run Detection</b> to see results, metrics, and analytics here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <div class="footer-stack">
            <span class="footer-chip">🐍 Python</span>
            <span class="footer-chip">👁️ OpenCV</span>
            <span class="footer-chip">🎈 Streamlit</span>
            <span class="footer-chip">🧠 Deep Learning</span>
        </div>
        <p class="footer-version">Version 1.0.0</p>
        <p class="footer-copyright">© 2026 Age &amp; Gender Detection AI. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True,
)