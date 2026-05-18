import csv
import streamlit as st
import keras
import numpy as np
from pathlib import Path
from PIL import Image

# Paths and constants
MODEL_PATH = Path(__file__).resolve().parent / "handloom_model.h5"
GI_TAGS_PATH = Path(__file__).resolve().parent / "gi_tags.csv"
CLASS_NAMES = ["muga", "nuni", "pat"]
CLASS_TO_TAG_NAME = {
    "muga": "Muga Silk",
    "nuni": "Nuni Silk",
    "pat": "Pat Silk",
}


def load_gi_tags():
    if not GI_TAGS_PATH.exists():
        return {}
    with GI_TAGS_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return {row["attire_name"].strip().lower(): row for row in reader}


def get_gi_tag_info(predicted_class):
    gi_tags = load_gi_tags()
    tag_name = CLASS_TO_TAG_NAME.get(predicted_class)
    if not tag_name:
        return None
    return gi_tags.get(tag_name.lower())


def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return keras.models.load_model(str(MODEL_PATH))


@st.cache_resource
def get_model():
    return load_model()


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(image):
    model = get_model()
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_class = CLASS_NAMES[predicted_index]
    return predicted_class, confidence


# --- Streamlit UI ---
st.set_page_config(page_title="Handloom Verification", page_icon="🧵", layout="centered")

# Colorful theme and layout tweaks
st.markdown(
        """
        <style>
        :root{
            --accent1: #FF7EB3; /* pink */
            --accent2: #65D6FF; /* light blue */
            --accent3: #FFC66D; /* warm yellow */
            --bg1: radial-gradient(circle at 20% 20%, rgba(88, 126, 255, 0.35) 0%, transparent 32%), radial-gradient(circle at 80% 80%, rgba(43, 211, 255, 0.20) 0%, transparent 32%), linear-gradient(180deg, #061430 0%, #0e1f52 45%, #081333 100%);
        }
        /* force dark blue background on Streamlit root containers */
        html, body, .stApp, [data-testid="stAppViewContainer"], .reportview-container, .main {
            font-family: 'Segoe UI', Roboto, Arial, sans-serif;
            background: var(--bg1) !important;
            color: #edf4ff;
            min-height: 100vh;
        }
        .app-header{ background: linear-gradient(90deg, rgba(255,126,179,0.95), rgba(101,214,255,0.95)); color: white; padding:18px; border-radius:12px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); backdrop-filter: blur(6px); }
        .app-header h1{ margin:0; font-size:28px; }
        .app-sub{ opacity:0.95; margin-top:6px; }
        .css-1d391kg, [data-testid="stSidebar"] { background: rgba(7, 21, 59, 0.92) !important; color: #edf4ff !important; }
        .css-1d391kg * , [data-testid="stSidebar"] * { color: #edf4ff !important; }
        .css-1d391kg .css-18e3th9, [data-testid="stSidebar"] .css-18e3th9 { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.14) !important; }
        .css-1d391kg .css-10trblm, [data-testid="stSidebar"] .css-10trblm { color: #cfe2ff !important; }
        .result-box { background: rgba(255,255,255,0.16); padding: 14px; border-radius: 10px; box-shadow: 0 6px 12px rgba(0,0,0,0.06); border: 1px solid rgba(255,255,255,0.08); }
        .badge { display:inline-block; padding:8px 12px; border-radius:999px; color: #fff; font-weight:600; margin-top:6px; }
        .badge .conf{ opacity:0.9; margin-left:8px; font-weight:500; }
        .stButton>button{ background: linear-gradient(90deg,var(--accent2),var(--accent3)); color:white; border:none; }
        .stProgress>div>div{ background: linear-gradient(90deg,var(--accent1),var(--accent2)); }
        </style>

        <div class="app-header">
            <h1>🧵 AI GI Handloom Verification</h1>
            <div class="app-sub">Upload a fabric image or use your webcam to identify fabric type and view GI tag details.</div>
        </div>
        """,
        unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write("Simple interface to predict handloom class and view GI tag details.")
    st.caption("Tip: Use clear, well-lit photos for best results.")

mode = st.sidebar.selectbox("Mode", ["Upload Image", "Realtime (Webcam)"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Fabric Image", type=["jpg", "jpeg", "png"]) 

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Prediction and results
        with col2:
            st.markdown("## Prediction")
            if st.button("Predict"):
                with st.spinner("Running model..."):
                    result, confidence = predict_image(image)

                # Show key metrics
                st.metric("Detected Fabric", CLASS_TO_TAG_NAME.get(result, result))
                percent = confidence * 100
                st.progress(min(max(percent / 100.0, 0.0), 1.0))
                st.write(f"Confidence: {percent:.2f}%")

                # Color badge for detected class
                colors = {"muga": "var(--accent1)", "nuni": "var(--accent2)", "pat": "var(--accent3)"}
                color = colors.get(result, "var(--accent1)")
                badge_html = f"<div class='badge' style='background:{color}'>" + f"{CLASS_TO_TAG_NAME.get(result, result)} <span class='conf'>{percent:.1f}%</span>" + "</div>"
                st.markdown(badge_html, unsafe_allow_html=True)

                # GI tag details
                gi_info = get_gi_tag_info(result)
                if gi_info:
                    with st.expander("GI Tag Details", expanded=True):
                        st.write(f"**Tag Name:** {gi_info.get('attire_name', '—')}")
                        st.write(f"**GI Tag:** {gi_info.get('gi_tag', '—')}")
                        st.write(f"**State:** {gi_info.get('state', '—')}")
                        st.write(f"**Type:** {gi_info.get('type', '—')}")
                        st.write(f"**Description:** {gi_info.get('description', '—')}")
                else:
                    st.info("GI tag information is not available for this prediction.")

            else:
                st.info("Press Predict to run the model on the uploaded image.")

    else:
        st.info("Upload a JPG or PNG image to get started.")

else:
    st.info("Realtime webcam mode — capture images from your camera.")
    cam_image = st.camera_input("Use your webcam to capture an image")

    if cam_image is not None:
        image = Image.open(cam_image).convert('RGB')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Camera capture", use_column_width=True)

        with col2:
            st.markdown("## Prediction (Camera)")
            if st.button("Predict from Camera"):
                with st.spinner("Running model on camera capture..."):
                    result, confidence = predict_image(image)

                st.metric("Detected Fabric", CLASS_TO_TAG_NAME.get(result, result))
                percent = confidence * 100
                st.progress(min(max(percent / 100.0, 0.0), 1.0))
                st.write(f"Confidence: {percent:.2f}%")

                colors = {"muga": "var(--accent1)", "nuni": "var(--accent2)", "pat": "var(--accent3)"}
                color = colors.get(result, "var(--accent1)")
                badge_html = f"<div class='badge' style='background:{color}'>" + f"{CLASS_TO_TAG_NAME.get(result, result)} <span class='conf'>{percent:.1f}%</span>" + "</div>"
                st.markdown(badge_html, unsafe_allow_html=True)

                gi_info = get_gi_tag_info(result)
                if gi_info:
                    with st.expander("GI Tag Details", expanded=True):
                        st.write(f"**Tag Name:** {gi_info.get('attire_name', '—')}")
                        st.write(f"**GI Tag:** {gi_info.get('gi_tag', '—')}")
                        st.write(f"**State:** {gi_info.get('state', '—')}")
                        st.write(f"**Type:** {gi_info.get('type', '—')}")
                        st.write(f"**Description:** {gi_info.get('description', '—')}")
                else:
                    st.info("GI tag information is not available for this prediction.")

            else:
                st.info("Capture an image with the camera above, then press Predict.")
