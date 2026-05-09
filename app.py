import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_PATH = Path("../../handloom_model.h5")
model = tf.keras.models.load_model(str(MODEL_PATH))
# Classes
CLASS_NAMES = ["muga", "nuni", "pat"]


# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# Prediction function
def predict_image(image):
    processed = preprocess_image(image)

    prediction = model.predict(processed)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# -----------------------------
# STREAMLIT UI STARTS HERE
# -----------------------------

st.title("AI GI Handloom Verification System")

st.write("Upload a fabric image for prediction")

# THIS IS WHERE YOU WRITE IT
uploaded_file = st.file_uploader(
    "Upload Fabric Image",
    type=["jpg", "jpeg", "png"]
)

# When image is uploaded
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    result, confidence = predict_image(image)

    st.subheader("Prediction Result")

    st.write(f"Detected Fabric: {result}")

    st.write(f"Confidence: {confidence*100:.2f}%")