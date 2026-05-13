import csv
import streamlit as st
import keras
import numpy as np
from pathlib import Path
from PIL import Image

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


st.title("AI GI Handloom Verification System")
st.write("Upload a fabric image for prediction")

uploaded_file = st.file_uploader(
    "Upload Fabric Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    result, confidence = predict_image(image)
    st.subheader("Prediction Result")
    st.write(f"Detected Fabric: {result}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    gi_info = get_gi_tag_info(result)
    if gi_info:
        st.subheader("GI Tag Details")
        st.write(f"**Tag Name:** {gi_info['attire_name']}")
        st.write(f"**GI Tag:** {gi_info['gi_tag']}")
        st.write(f"**State:** {gi_info['state']}")
        st.write(f"**Type:** {gi_info['type']}")
        st.write(f"**Description:** {gi_info['description']}")
    else:
        st.warning("GI tag information is not available for this prediction.")
