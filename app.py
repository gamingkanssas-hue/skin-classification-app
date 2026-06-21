"""
==========================================================
 Klasifikasi Permasalahan Kulit Wajah — Aplikasi Web (Streamlit)
 Menggunakan Ekstraksi Fitur Warna & Tekstur + SVM
----------------------------------------------------------
 Pipeline ekstraksi fitur PERSIS SAMA dengan notebook training:
   - Preprocessing: resize 224x224, konversi RGB / HSV / Grayscale
   - Fitur Warna  (36): RGB mean/std + RGB histogram 8-bin + HSV mean/std
   - Fitur Tekstur(14): GLCM 4 arah (contrast, correlation, energy,
                        homogeneity) + LBP histogram 10-bin
   - Total: 50 dimensi -> StandardScaler -> SVM (kernel terbaik dari
     GridSearchCV) -> prediksi 6 kelas
==========================================================
"""

import os
import pickle
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from features import extract_features_pipeline, IMG_SIZE

# ----------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------
st.set_page_config(
    page_title="Klasifikasi Kulit Wajah | SVM",
    page_icon="🩷",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "saved_model"

#warna
COLOR_MAROON      = "#8c1c2b"
COLOR_MAROON_DARK = "#5e1019"
COLOR_PINK        = "#e8a0ab"
COLOR_PINK_SOFT   = "#f6d9de"
COLOR_CREAM       = "#fdf6f3"
COLOR_CREAM2      = "#fbeae9"

CLASS_INFO = {
    "acne": {
        "label_id": "Jerawat (Acne)",
        "desc": "Terdeteksi pola peradangan kulit yang umum dikaitkan dengan jerawat.",
        "icon": "🔴",
    },
    "dark spots": {
        "label_id": "Bintik Hitam (Dark Spots)",
        "desc": "Terdeteksi area hiperpigmentasi / bintik hitam pada kulit.",
        "icon": "🟤",
    },
    "Redness": {
        "label_id": "Kemerahan (Redness)",
        "desc": "Terdeteksi area kemerahan pada permukaan kulit.",
        "icon": "🌹",
    },
    "pores": {
        "label_id": "Pori-Pori Membesar (Pores)",
        "desc": "Terdeteksi pola pori-pori kulit yang membesar.",
        "icon": "🕳️",
    },
    "wrinkles": {
        "label_id": "Kerutan (Wrinkles)",
        "desc": "Terdeteksi garis halus / kerutan pada permukaan kulit.",
        "icon": "〰️",
    },
    "normal": {
        "label_id": "Kulit Normal",
        "desc": "Tidak ditemukan indikasi permasalahan kulit yang signifikan.",
        "icon": "✅",
    },
}

# ----------------------------------------------------------
# CSS — tema mengikuti poster (maroon / pink / cream)
# ----------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, {COLOR_CREAM} 0%, {COLOR_CREAM2} 100%);
    }}

    /* Header banner mirip poster */
    .hero-banner {{
        background: linear-gradient(135deg, {COLOR_MAROON_DARK} 0%, {COLOR_MAROON} 55%, {COLOR_PINK} 130%);
        border-radius: 18px;
        padding: 2.1rem 2rem 1.7rem 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(140,28,43,0.25);
    }}
    .hero-banner h1 {{
        color: #ffffff;
        font-weight: 800;
        font-size: 2.1rem;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
        text-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    .hero-banner p {{
        color: #f6d9de;
        font-size: 1rem;
        margin: 0;
    }}
    .hero-pill {{
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        color: white;
        padding: 3px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-top: 0.7rem;
        margin-right: 6px;
    }}

    /* Section cards seperti kartu di poster */
    .section-card {{
        background: #ffffff;
        border: 1px solid {COLOR_PINK_SOFT};
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 4px 14px rgba(140,28,43,0.06);
        margin-bottom: 1.1rem;
    }}
    .section-title {{
        display: flex;
        align-items: center;
        gap: 8px;
        background: {COLOR_MAROON};
        color: white;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.92rem;
        width: fit-content;
        margin-bottom: 0.9rem;
    }}

    /* Hasil prediksi utama */
    .result-box {{
        background: linear-gradient(135deg, {COLOR_MAROON} 0%, {COLOR_MAROON_DARK} 100%);
        color: white;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(140,28,43,0.3);
    }}
    .result-box h2 {{
        margin: 0.2rem 0;
        font-size: 1.6rem;
        font-weight: 800;
    }}
    .result-box .conf {{
        font-size: 1rem;
        opacity: 0.92;
    }}

    /* Bar probabilitas custom */
    .prob-row {{
        display: flex;
        align-items: center;
        margin-bottom: 7px;
        gap: 10px;
    }}
    .prob-label {{
        width: 165px;
        font-size: 0.85rem;
        color: {COLOR_MAROON_DARK};
        font-weight: 600;
    }}
    .prob-track {{
        flex: 1;
        background: {COLOR_PINK_SOFT};
        border-radius: 999px;
        height: 14px;
        overflow: hidden;
    }}
    .prob-fill {{
        height: 100%;
        border-radius: 999px;
    }}
    .prob-pct {{
        width: 52px;
        font-size: 0.82rem;
        font-weight: 700;
        color: {COLOR_MAROON_DARK};
        text-align: right;
    }}

    /* Tombol */
    .stButton>button, .stDownloadButton>button {{
        background: {COLOR_MAROON};
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        background: {COLOR_MAROON_DARK};
        color: white;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        color: {COLOR_MAROON_DARK};
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLOR_MAROON} !important;
        border-bottom-color: {COLOR_MAROON} !important;
    }}

    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}

    .footer-note {{
        text-align: center;
        color: #9c6b72;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# LOAD MODEL (cached)
# ----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    required = ["model_svm.pkl", "scaler.pkl", "classes.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        return None, None, None, None, None

    with open(os.path.join(MODEL_DIR, "model_svm.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "classes.pkl"), "rb") as f:
        classes = pickle.load(f)

    best_params, metrics = None, None
    bp_path = os.path.join(MODEL_DIR, "best_params.pkl")
    mt_path = os.path.join(MODEL_DIR, "metrics.pkl")
    if os.path.exists(bp_path):
        with open(bp_path, "rb") as f:
            best_params = pickle.load(f)
    if os.path.exists(mt_path):
        with open(mt_path, "rb") as f:
            metrics = pickle.load(f)

    return model, scaler, classes, best_params, metrics


model, scaler, CLASSES, best_params, metrics = load_model()

CLASS_COLORS = ["#e74c3c", "#8e44ad", "#e67e22", "#2980b9", "#16a085", "#27ae60"]


# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="background:{COLOR_MAROON};padding:1rem;border-radius:12px;margin-bottom:1rem;">
        <h3 style="color:white;margin:0;">🩷 Tentang Aplikasi</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
**Klasifikasi Permasalahan Kulit Wajah**
menggunakan ekstraksi fitur warna & tekstur
(RGB, HSV, GLCM, LBP) dengan algoritma **SVM**.
    """)

    if model is not None:
        st.success("✅ Model berhasil dimuat")
        if best_params:
            st.markdown("**⚙️ Parameter Model Terbaik**")
            st.write(f"- Kernel : `{best_params.get('kernel','-')}`")
            st.write(f"- C : `{best_params.get('C','-')}`")
            st.write(f"- Gamma : `{best_params.get('gamma','-')}`")
        if metrics:
            st.markdown("**📊 Performa Model (Data Uji)**")
            st.write(f"- Akurasi : `{metrics.get('accuracy',0)*100:.2f}%`")
            st.write(f"- Precision : `{metrics.get('precision',0)*100:.2f}%`")
            st.write(f"- Recall : `{metrics.get('recall',0)*100:.2f}%`")
            st.write(f"- F1-Score : `{metrics.get('f1',0)*100:.2f}%`")
    else:
        st.error("❌ Model belum ditemukan")
        st.caption("Letakkan file .pkl di folder `saved_model/` — lihat README.")

    st.markdown("---")
    st.markdown("""
**🏷️ Kelas yang Dideteksi**
- 🔴 Acne (Jerawat)
- 🟤 Dark Spots
- 🌹 Redness
- 🕳️ Pores
- 〰️ Wrinkles
- ✅ Normal
    """)

    st.markdown("---")
    st.caption("👩‍🎓 Putu Eka Febriani (E1E124013)")
    st.caption("👨‍🎓 Farid Khandra (E1E124006)")
    st.caption("👨‍🏫 Pembimbing: Rizal Adi Saputra, S.T., M.T")


# ----------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <h1>🩷 KLASIFIKASI PERMASALAHAN KULIT WAJAH</h1>
    <p>Ekstraksi Fitur Warna & Tekstur Berbasis Machine Learning dengan Algoritma <b>Support Vector Machine (SVM)</b></p>
    <span class="hero-pill">📷 Upload / Scan Foto</span>
    <span class="hero-pill">🎨 Fitur Warna RGB · HSV</span>
    <span class="hero-pill">🧬 Fitur Tekstur GLCM · LBP</span>
    <span class="hero-pill">🤖 SVM Classifier</span>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning(
        "**Model belum tersedia.** Upload file `model_svm.pkl`, `scaler.pkl`, dan "
        "`classes.pkl` ke folder `saved_model/` pada repository, lalu deploy ulang. "
        "Lihat bagian README untuk instruksi lengkap."
    )
    st.stop()


# ----------------------------------------------------------
# TABS: Upload File  |  Scan Kamera
# ----------------------------------------------------------
tab_upload, tab_camera = st.tabs(["📁 Upload Foto", "📸 Scan Langsung (Kamera)"])

image_source = None

with tab_upload:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📁 Upload Foto Kulit Wajah</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih gambar wajah (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="uploader",
    )
    st.caption("💡 Tips: gunakan foto close-up area kulit dengan pencahayaan cukup, tanpa makeup berat, untuk hasil terbaik.")
    st.markdown('</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file).convert("RGB")

with tab_camera:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📸 Ambil Foto Langsung dari Kamera</div>', unsafe_allow_html=True)
    camera_file = st.camera_input("Arahkan kamera ke area kulit wajah, lalu ambil gambar")
    st.caption("💡 Pastikan wajah berada di tengah frame dan pencahayaan merata.")
    st.markdown('</div>', unsafe_allow_html=True)
    if camera_file is not None:
        image_source = Image.open(camera_file).convert("RGB")


# ----------------------------------------------------------
# PROSES & HASIL PREDIKSI
# ----------------------------------------------------------
if image_source is not None:
    col_img, col_result = st.columns([1, 1.3], gap="large")

    # Convert PIL -> array RGB -> simpan sementara utk pipeline (pakai cv2 BGR)
    img_rgb_full = np.array(image_source)
    img_bgr_full = cv2.cvtColor(img_rgb_full, cv2.COLOR_RGB2BGR)

    with col_img:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🖼️ Gambar Input</div>', unsafe_allow_html=True)
        st.image(image_source, use_container_width=True, caption="Gambar yang akan dianalisis")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🔍 Mengekstraksi fitur warna & tekstur, lalu mengklasifikasikan..."):
        try:
            feature_vector = extract_features_pipeline(img_bgr_full)
        except Exception as e:
            feature_vector = None
            st.error(f"Terjadi kesalahan saat ekstraksi fitur: {e}")

    if feature_vector is not None:
        vec_scaled = scaler.transform([feature_vector])
        pred_idx = model.predict(vec_scaled)[0]
        pred_label = CLASSES[pred_idx]
        info = CLASS_INFO.get(pred_label, {"label_id": pred_label, "desc": "", "icon": "🔍"})

        if hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(vec_scaled)[0]
        else:
            pred_proba = np.zeros(len(CLASSES))
            pred_proba[pred_idx] = 1.0

        with col_result:
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:2.4rem;">{info['icon']}</div>
                <h2>{info['label_id']}</h2>
                <div class="conf">Tingkat Keyakinan: <b>{max(pred_proba)*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown(f"<div style='color:{COLOR_MAROON_DARK};'>{info['desc']}</div>", unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Confidence Score per Kelas</div>', unsafe_allow_html=True)

            sorted_results = sorted(
                zip(CLASSES, pred_proba, range(len(CLASSES))),
                key=lambda x: -x[1]
            )
            for cls_name, prob, orig_idx in sorted_results:
                label_disp = CLASS_INFO.get(cls_name, {}).get("label_id", cls_name)
                bar_color = COLOR_MAROON if cls_name == pred_label else COLOR_PINK
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-label">{label_disp}</div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{prob*100:.1f}%; background:{bar_color};"></div>
                    </div>
                    <div class="prob-pct">{prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            "ℹ️ **Disclaimer:** Hasil ini merupakan prediksi otomatis dari model machine learning "
            "untuk keperluan akademik/penelitian, **bukan diagnosis medis**. Untuk penanganan kulit "
            "yang akurat, konsultasikan dengan dokter kulit / dermatolog profesional."
        )
else:
    st.markdown(f"""
    <div style="text-align:center; padding:2.5rem 1rem; color:{COLOR_MAROON_DARK};">
        <div style="font-size:2.6rem;">⬆️</div>
        <p style="font-size:1rem;">Silakan <b>upload foto</b> atau <b>scan langsung via kamera</b> di atas untuk memulai analisis.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    🩷 Klasifikasi Permasalahan Kulit Wajah — SVM &nbsp;|&nbsp; Dibuat untuk keperluan akademik
</div>
""", unsafe_allow_html=True)
