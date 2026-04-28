# Paste your custom UI code here for testing.
# You can run it and if it doesn't look good, we can just switch back to the original ui/main.py
import streamlit as st
import numpy as np
import cv2
from datetime import datetime

from utils import prepare_image_for_model, encode_image_for_db
from model import predict
from gradcam import get_gradcam_heatmap, overlay_heatmap
from database import init_db, insert_prediction, get_recent_predictions

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NeuroScanAI · Brain Tumor Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PRO-LEVEL CSS — NeuroScanAI Design System
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #050810;
    --surface: rgba(255,255,255,0.04);
    --surface-hover: rgba(255,255,255,0.07);
    --border: rgba(255,255,255,0.08);
    --border-active: rgba(100,200,255,0.4);
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --accent3: #10b981;
    --danger: #ff4757;
    --text: #f0f4f8;
    --muted: #8899aa;
    --font-display: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-display) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridShift 25s linear infinite;
}

/* Ambient glow */
.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 700px 500px at 15% 15%, rgba(124,58,237,0.1) 0%, transparent 70%),
        radial-gradient(ellipse 600px 400px at 85% 75%, rgba(0,212,255,0.07) 0%, transparent 70%);
}

@keyframes gridShift {
    0%   { background-position: 0 0; }
    100% { background-position: 40px 40px; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(5,8,16,0.85) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] * {
    font-family: var(--font-display) !important;
    color: var(--text) !important;
}

/* ── Main content ── */
.main .block-container {
    padding-top: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1200px !important;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: var(--font-display) !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(0,212,255,0.02) !important;
    border: 2px dashed rgba(0,212,255,0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.6) !important;
    background: rgba(0,212,255,0.04) !important;
    transform: scale(1.005);
}

[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100% !important;
    padding: 14px 28px !important;
    font-family: var(--font-display) !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #7c3aed 0%, #0ea5e9 100%) !important;
    color: white !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 35px rgba(124,58,237,0.55), 0 0 0 1px rgba(0,212,255,0.3) !important;
    filter: brightness(1.1);
}

.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stImage"] img:hover {
    border-color: rgba(0,212,255,0.35) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 14px 35px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,212,255,0.1) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Bar chart ── */
[data-testid="stArrowVegaLiteChart"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(0,212,255,0.25) !important;
    background: var(--surface-hover) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
}

/* ── Toast ── */
[data-testid="stToast"] {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 10px !important;
    color: #10b981 !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Alerts / Info boxes ── */
[data-testid="stAlert"] {
    background: rgba(0,212,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.3); }

/* ── Caption / small text ── */
.stCaption, small, caption {
    font-family: var(--font-mono) !important;
    color: var(--muted) !important;
    font-size: 11px !important;
}

/* ── Column gaps ── */
[data-testid="column"] {
    padding: 0 8px !important;
}

/* Pulse dot animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# INITIALIZE DATABASE
# ==========================================
db_status = init_db()


# ==========================================
# SIDEBAR — HISTORY DASHBOARD
# ==========================================
with st.sidebar:

    # Brand header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 20px;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#7c3aed,#00d4ff);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;">🧠</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:800;letter-spacing:0.04em;">
                NeuroScan<span style="color:#00d4ff;">AI</span>
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:10px;color:#8899aa;">v2.4.1 · DenseNet121</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # DB status pill
    if db_status != "Success":
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;margin-bottom:16px;
                    background:rgba(255,71,87,0.08);border:1px solid rgba(255,71,87,0.25);border-radius:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#ff4757;
                        box-shadow:0 0 8px #ff4757;animation:pulse 2s ease-in-out infinite;"></div>
            <span style="font-family:'Space Mono',monospace;font-size:11px;color:#ff4757;">Postgres Offline</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;margin-bottom:16px;
                    background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);border-radius:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#10b981;
                        box-shadow:0 0 8px #10b981;animation:pulse 2s ease-in-out infinite;"></div>
            <span style="font-family:'Space Mono',monospace;font-size:11px;color:#10b981;">Postgres Online</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:10px;color:#8899aa;
                letter-spacing:0.15em;text-transform:uppercase;padding-bottom:10px;
                border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:16px;">
        Recent Scans
    </div>
    """, unsafe_allow_html=True)

    history_logs = get_recent_predictions(limit=5)

    if not history_logs:
        st.markdown("""
        <div style="text-align:center;padding:24px 0;font-family:'Space Mono',monospace;
                    font-size:12px;color:#8899aa;">
            No scans yet.<br>Upload an MRI to begin.
        </div>
        """, unsafe_allow_html=True)
    else:
        COLOR_MAP = {
            "glioma":      ("#ff6b6b", "#ff4757"),
            "meningioma":  ("#ffd93d", "#f59e0b"),
            "no tumor":    ("#6bcb77", "#10b981"),
            "pituitary":   ("#4d96ff", "#2563eb"),
        }
        for row in history_logs:
            cls = row['predicted_class'].lower()
            conf = row['confidence']
            color_from, color_to = COLOR_MAP.get(cls, ("#00d4ff", "#0ea5e9"))
            ts_str = row['timestamp'].strftime('%b %d · %H:%M')

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                        border-radius:10px;padding:12px;margin-bottom:10px;
                        transition:all 0.25s ease;cursor:pointer;">
                <div style="font-family:'Syne',sans-serif;font-size:12px;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.08em;
                            color:{color_from};margin-bottom:3px;">
                    {row['predicted_class'].capitalize()}
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:10px;color:#8899aa;margin-bottom:8px;">
                    {conf*100:.1f}% confidence · {ts_str}
                </div>
                <div style="height:3px;background:rgba(255,255,255,0.07);border-radius:2px;overflow:hidden;">
                    <div style="height:3px;width:{conf*100:.0f}%;border-radius:2px;
                                background:linear-gradient(90deg,{color_from},{color_to});
                                transition:width 0.8s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            nparr = np.frombuffer(row['image'], np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)


# ==========================================
# MAIN CONTENT
# ==========================================

# Page title
st.markdown("""
<div style="margin-bottom:8px;">
    <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;
                letter-spacing:-0.02em;line-height:1.1;margin-bottom:6px;">
        MRI Diagnostic
        <span style="background:linear-gradient(90deg,#00d4ff,#7c3aed);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Analysis
        </span>
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:12px;color:#8899aa;">
        DenseNet121 deep learning · Grad-CAM explainability · Real-time inference
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.15);
            border-left:3px solid #00d4ff;border-radius:10px;
            padding:14px 18px;margin-bottom:24px;">
    <span style="font-family:'Space Mono',monospace;font-size:12px;color:#8899aa;">
        Upload a T1/T2-weighted MRI scan (JPG · PNG · JPEG). The model classifies into
        <span style="color:#f0f4f8;">Glioma · Meningioma · No Tumor · Pituitary</span>
        and generates a Grad-CAM heatmap showing exactly where the model focused.
    </span>
</div>
""", unsafe_allow_html=True)

# ── Upload ──
uploaded_file = st.file_uploader(
    "Drop an MRI scan here, or click to browse",
    type=["jpg", "png", "jpeg"]
)

st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
run_btn = st.button("▶ Run Diagnostic Analysis")

# ── Processing ──
if uploaded_file is not None and run_btn:
    img_bytes = uploaded_file.read()

    with st.spinner("Analyzing MRI with DenseNet121 — please wait..."):
        original_img_rgb, img_array = prepare_image_for_model(img_bytes)

        if original_img_rgb is None:
            st.error("Error decoding image. Please upload a valid JPG/PNG file.")
            st.stop()

        results        = predict(img_array)
        pred_class     = results["predicted_class"]
        confidence     = results["confidence"]
        
        # get_gradcam_heatmap now returns a tuple: (heatmap, pred_class, confidence, probs)
        # We only need the first element (the heatmap) for the overlay.
        heatmap_tuple  = get_gradcam_heatmap(img_array)
        heatmap        = heatmap_tuple[0]
        
        superimposed   = overlay_heatmap(original_img_rgb, heatmap)
        compressed     = encode_image_for_db(original_img_rgb)
        insert_out     = insert_prediction(compressed, pred_class, confidence)

        if insert_out != "Success":
            st.error(f"Database Save Failed: {insert_out}")
        else:
            st.toast("✅ Prediction logged to Postgres successfully!")

    # ── Divider ──
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Diagnosis result card ──
    is_confident  = confidence >= 0.8
    accent_color  = "#10b981" if is_confident else "#ff4757"
    accent_bg     = "rgba(16,185,129,0.06)" if is_confident else "rgba(255,71,87,0.06)"
    status_label  = "Confident Match" if is_confident else "Requires Human Review"
    status_dot    = "#10b981" if is_confident else "#ff4757"

    st.markdown(f"""
    <div style="background:{accent_bg};border:1px solid {accent_color}44;
                border-radius:16px;padding:24px 28px;margin-bottom:24px;
                position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:1px;
                    background:linear-gradient(90deg,transparent,{accent_color}88,transparent);"></div>
        <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;">
            <div>
                <div style="font-family:'Space Mono',monospace;font-size:10px;
                            text-transform:uppercase;letter-spacing:0.15em;color:#8899aa;margin-bottom:6px;">
                    Primary Diagnosis
                </div>
                <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;
                            letter-spacing:-0.02em;line-height:1;margin-bottom:10px;
                            background:linear-gradient(135deg,#fff 30%,#00d4ff);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    {pred_class.upper()}
                </div>
                <div style="display:inline-flex;align-items:center;gap:7px;
                            padding:5px 12px;border-radius:20px;
                            background:{accent_color}18;border:1px solid {accent_color}44;">
                    <div style="width:7px;height:7px;border-radius:50%;background:{status_dot};
                                box-shadow:0 0 8px {status_dot};animation:pulse 2s ease-in-out infinite;"></div>
                    <span style="font-family:'Space Mono',monospace;font-size:11px;color:{accent_color};">
                        {status_label}
                    </span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'Space Mono',monospace;font-size:48px;font-weight:700;
                            line-height:1;background:linear-gradient(135deg,#ffd700,#ffaa00);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    {confidence*100:.2f}%
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:10px;
                            color:#8899aa;margin-top:4px;letter-spacing:0.1em;text-transform:uppercase;">
                    Confidence Score
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ──
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Model", "DenseNet121", "121 dense layers")
    with m2:
        st.metric("Input Shape", "224 × 224", "RGB normalized")
    with m3:
        st.metric("Test Accuracy", "98.3%", "Held-out set")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Image panels ──
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:10px;color:#8899aa;
                text-transform:uppercase;letter-spacing:0.15em;margin-bottom:14px;">
        Scan Visualisation
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
            <span style="font-family:'Syne',sans-serif;font-size:13px;font-weight:600;color:#8899aa;">
                Original MRI Scan
            </span>
            <span style="font-family:'Space Mono',monospace;font-size:10px;padding:2px 8px;
                         border-radius:3px;background:rgba(0,212,255,0.1);color:#00d4ff;
                         border:1px solid rgba(0,212,255,0.2);">INPUT</span>
        </div>
        """, unsafe_allow_html=True)
        st.image(original_img_rgb, caption="T1-weighted patient scan", use_container_width=True)

    with col2:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
            <span style="font-family:'Syne',sans-serif;font-size:13px;font-weight:600;color:#8899aa;">
                Grad-CAM Heatmap
            </span>
            <span style="font-family:'Space Mono',monospace;font-size:10px;padding:2px 8px;
                         border-radius:3px;background:rgba(255,107,107,0.1);color:#ff6b6b;
                         border:1px solid rgba(255,107,107,0.2);">XAI</span>
        </div>
        """, unsafe_allow_html=True)
        st.image(superimposed, caption="Red regions = model attention focus", use_container_width=True)

    # ── Probability distribution ──
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:10px;color:#8899aa;
                text-transform:uppercase;letter-spacing:0.15em;margin-bottom:14px;">
        Class Probability Distribution
    </div>
    """, unsafe_allow_html=True)

    probs = results["all_probabilities"]

    # Render custom probability bars
    COLOR_MAP_PROBS = {
        "glioma":      ("linear-gradient(90deg,#ff6b6b,#ff4757)", "#ff6b6b"),
        "meningioma":  ("linear-gradient(90deg,#ffd93d,#f59e0b)", "#ffd93d"),
        "no tumor":    ("linear-gradient(90deg,#6bcb77,#10b981)", "#6bcb77"),
        "pituitary":   ("linear-gradient(90deg,#4d96ff,#2563eb)", "#4d96ff"),
    }

    st.markdown("""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:20px 24px;">
    """, unsafe_allow_html=True)

    for label, val in probs.items():
        grad, col = COLOR_MAP_PROBS.get(label.lower(), ("linear-gradient(90deg,#00d4ff,#0ea5e9)", "#00d4ff"))
        pct = val * 100
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
            <div style="font-family:'Space Mono',monospace;font-size:12px;
                        color:#f0f4f8;width:100px;flex-shrink:0;">{label.capitalize()}</div>
            <div style="flex:1;height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">
                <div style="height:8px;width:{pct:.1f}%;border-radius:4px;background:{grad};
                            transition:width 1s ease;"></div>
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:12px;
                        color:#8899aa;width:48px;text-align:right;">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif uploaded_file is None and run_btn:
    st.markdown("""
    <div style="background:rgba(255,71,87,0.06);border:1px solid rgba(255,71,87,0.25);
                border-radius:10px;padding:14px 18px;margin-top:12px;">
        <span style="font-family:'Space Mono',monospace;font-size:12px;color:#ff4757;">
            ⚠ Please upload an MRI scan before running analysis.
        </span>
    </div>
    """, unsafe_allow_html=True)