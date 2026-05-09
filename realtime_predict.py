import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import argparse

# Model and classes
MODEL_PATH = "handloom_model.h5"
CLASSES = ["muga", "nuni", "pat"]

def preprocess_frame(frame):
    """Preprocess the frame for model prediction"""
    # Resize to match model input
    img = cv2.resize(frame, (224, 224))
    # Convert to float32 and normalize
    img = img.astype("float32") / 255.0
    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)
    return img

def predict_on_image(image_path):
    """Predict on a single image file"""
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Cannot load image: {image_path}")
        return

    # Preprocess and predict
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame, verbose=0)[0]
    predicted_class_idx = np.argmax(prediction)
    predicted_class = CLASSES[predicted_class_idx]
    confidence = float(prediction[predicted_class_idx]) * 100

    # Display result
    text = f"{predicted_class}: {confidence:.1f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Handloom Cloth Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"🎯 Prediction: {predicted_class} ({confidence:.1f}% confidence)")

def main():
    parser = argparse.ArgumentParser(description="Real-time handloom cloth detection")
    parser.add_argument('--image', type=str, help='Path to image file for prediction')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    args = parser.parse_args()

    # If image path provided, use image mode
    if args.image:
        predict_on_image(args.image)
        return

    # Load the model
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ Cannot open camera. Try using --image <path> instead.")
        print("Available options:")
        print("  python realtime_predict.py --image test.jpg")
        print("  python realtime_predict.py --camera 1  (try different camera index)")
        sys.exit(1)

    print("📹 Camera opened. Press 'q' to quit.")
    print("🎯 Point camera at handloom cloth for detection.")
    print("💡 Tip: Hold the cloth steady for better detection.")

    frame_count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame. Retrying...")
            cv2.waitKey(100)
            continue

        frame_count += 1

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make prediction every 5 frames to reduce lag
        if frame_count % 5 == 0:
            try:
                prediction = model.predict(processed_frame, verbose=0)[0]
                predicted_class_idx = np.argmax(prediction)
                predicted_class = CLASSES[predicted_class_idx]
                confidence = float(prediction[predicted_class_idx]) * 100
            except Exception as e:
                predicted_class = "Error"
                confidence = 0.0

        # Display the prediction on the frame
        text = f"{predicted_class}: {confidence:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Show the frame
        cv2.imshow('Handloom Cloth Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera closed.")

if __name__ == "__main__":
    main()
