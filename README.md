# 🧠 Age & Gender Detection AI — Dashboard

A professional, portfolio-ready **AI Dashboard** for real-time face detection with age range and gender prediction, built with **Python**, **OpenCV DNN**, **Streamlit**, and **Plotly**.

![Status](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.38-red)

---

## ✨ Overview

This project detects human faces in an uploaded image or live camera capture, then predicts each face's **age range** and **gender** along with confidence scores — all presented in a modern, glassmorphism-inspired dashboard UI instead of a default Streamlit layout.

> **Note:** This redesign only touches the frontend (`app.py` + `assets/style.css`). All detection logic remains in `detector.py` via the existing `process_image()` function and is used unmodified.

---

## 🖥️ Features

- 🎨 Premium hero banner with custom branding
- 🧭 Rich sidebar (About, Features, Tech Stack, Model Info, Developer)
- 📤 Beautiful upload card with drag-and-drop **or** live camera input
- 🚀 Large, animated, rounded detection button
- 🖼️ Side-by-side Original vs Processed image comparison cards
- 📊 Live metric cards — Total Faces, Male Count, Female Count, Processing Time, Timestamp
- 📋 Professional results table (per-face gender, age, and confidence breakdown)
- 📈 Interactive Plotly charts:
  - Gender Confidence Bar Chart
  - Age Confidence Bar Chart
  - Male vs Female Pie Chart
  - Face Detection Confidence Chart
- ⬇️ One-click download of the processed image
- 🎨 Custom CSS theme — soft shadows, rounded cards, blue palette, hover animations
- 📱 Responsive, clean, and recruiter/GitHub/LinkedIn-ready presentation

---

## 📁 Project Structure

```
Age_Gender_Detection/
│
├── app.py                 # Streamlit dashboard (frontend only — redesigned)
├── detector.py             # Existing detection logic (unmodified)
├── requirements.txt
├── README.md
│
├── assets/
│   ├── style.css           # Custom dashboard theme
│   ├── logo.png             # Project logo (optional — add your own)
│   ├── banner.png           # Hero banner background (optional)
│   └── background.png       # Extra background asset (optional)
│
├── models/                 # DNN model weights used by detector.py
├── output/                 # Saved processed outputs
└── temp/                   # Temporary working files
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Age_Gender_Detection
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your model files** to the `models/` folder (as expected by your existing `detector.py`).

5. **Add branding assets (optional)** — place `logo.png` and `banner.png` inside `assets/`. If they're missing, the dashboard gracefully falls back to emoji-based branding.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`).

---

## 🔌 How the UI Connects to `detector.py`

The dashboard calls your existing function directly:

```python
processed_image, results, processing_time, timestamp = process_image(cv_image)
```

Where `results` is expected as:

```python
[
    {
        "gender": "Male",
        "gender_conf": 0.96,
        "age": "25-32",
        "age_conf": 0.84,
        "face_conf": 0.98
    },
    ...
]
```

No changes were made to this contract — the UI simply consumes it and renders it beautifully.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Face/Age/Gender Detection | OpenCV DNN Module |
| Web Framework | Streamlit |
| Navigation | streamlit-option-menu |
| UI Enhancements | streamlit-extras |
| Charts | Plotly |
| Data Handling | Pandas |
| Image Handling | Pillow, OpenCV |

---

## 📸 Screenshots

Add your own screenshots here after running the app, e.g.:

```
assets/screenshot-hero.png
assets/screenshot-results.png
```

---

## 👤 Developer

Built as an academic / portfolio project demonstrating applied computer vision, model integration, and full-stack ML dashboard deployment.

---

## 📄 License

This project is available for educational and portfolio use. Add your preferred license (MIT, Apache 2.0, etc.) here.

---

**Version:** 1.0.0
**© 2026 Age & Gender Detection AI. All rights reserved.**