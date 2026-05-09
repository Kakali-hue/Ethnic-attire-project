import argparse
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = Path("handloom_model.h5")
CLASS_NAMES = ["muga", "nuni", "pat"]


def preprocess_image(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict handloom class for a single image")
    parser.add_argument("image", type=Path, help="Path to the image file")
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = load_model(str(MODEL_PATH))
    img = preprocess_image(args.image)
    prediction = model.predict(img, verbose=0)[0]
    top_index = int(np.argmax(prediction))
    class_name = CLASS_NAMES[top_index]
    confidence = float(prediction[top_index]) * 100

    print(f"Prediction: {class_name} ({confidence:.2f}% confidence)")


if __name__ == "__main__":
    main()
