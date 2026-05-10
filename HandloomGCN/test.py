from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Load model
model = load_model("handloom_model.h5")

# Classes
classes = ["muga", "nuni", "pat"]

# Image path
img_path = "E:/Handloomproject/test.jpg"

# Check if file exists
if not os.path.exists(img_path):
    print("❌ Image not found at:", img_path)
    exit()

# Read image
img = cv2.imread(img_path)

if img is None:
    print("❌ Image could not be loaded. Check file format.")
    exit()

# Process image
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.reshape(img, (1,224,224,3))

# Predict
prediction = model.predict(img)
result = classes[np.argmax(prediction)]

print("✅ Prediction:", result)