from pathlib import Path

split_code = '''import random
import shutil
import zipfile
from pathlib import Path

BASE = Path("E:/Handloomproject/HandloomGCN")
ZIP_PATH = BASE / "new classes.zip"
EXTRACT_ROOT = BASE / "Our_Cropped_Handloom_Dataset"
DATASET_ROOT = BASE / "dataset"
TRAIN_ROOT = DATASET_ROOT / "train"
TEST_ROOT = DATASET_ROOT / "test"
CLASS_NAMES = ["muga", "nuni", "pat"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def extract_zip():
    if not ZIP_PATH.exists():
        return None

    print(f"📦 Extracting dataset ZIP: {ZIP_PATH}")
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_ROOT)
    return EXTRACT_ROOT


def find_class_root(root: Path):
    if not root.exists():
        return None

    candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
    for candidate in candidates:
        if all((candidate / cls).is_dir() for cls in CLASS_NAMES):
            return candidate

    for candidate in root.rglob("*"):
        if candidate.is_dir() and all((candidate / cls).is_dir() for cls in CLASS_NAMES):
            return candidate

    return None


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    return sorted([p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def split_folder(source: Path, train_dest: Path, test_dest: Path):
    images = list_images(source)
    random.shuffle(images)
    split_index = int(TRAIN_RATIO * len(images))
    train_files = images[:split_index]
    test_files = images[split_index:]

    for image_name in train_files:
        shutil.copy2(source / image_name, train_dest / image_name)
    for image_name in test_files:
        shutil.copy2(source / image_name, test_dest / image_name)

    print(f"  {source.name}: {len(train_files)} train, {len(test_files)} test")


def main():
    random.seed(RANDOM_SEED)

    source_root = extract_zip()
    if source_root is None:
        source_root = BASE

    class_root = find_class_root(source_root)
    if class_root is None:
        class_root = find_class_root(BASE)
    if class_root is None:
        raise FileNotFoundError(f"Could not find class folders {CLASS_NAMES} under {source_root} or {BASE}.")

    print(f"✅ Source folder found: {class_root}")

    ensure_clean_dir(TRAIN_ROOT)
    ensure_clean_dir(TEST_ROOT)

    for cls_name in CLASS_NAMES:
        source_cls = class_root / cls_name
        train_cls = TRAIN_ROOT / cls_name
        test_cls = TEST_ROOT / cls_name
        train_cls.mkdir(parents=True, exist_ok=True)
        test_cls.mkdir(parents=True, exist_ok=True)

        if not source_cls.exists():
            print(f"⚠️  Missing class source: {source_cls}")
            continue

        split_folder(source_cls, train_cls, test_cls)

    print("\n✅ Data split completed!")
    print(f"Train path: {TRAIN_ROOT}")
    print(f"Test path: {TEST_ROOT}")


if __name__ == "__main__":
    main()
'''

train_code = '''from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_PATH = DATASET_DIR / "train"
TEST_PATH = DATASET_DIR / "test"
MODEL_PATH = BASE_DIR.parent / "handloom_model.h5"
CLASS_NAMES = ["muga", "nuni", "pat"]

train_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)

test_gen = ImageDataGenerator(rescale=1.0 / 255.0)

train_data = train_gen.flow_from_directory(
    directory=str(TRAIN_PATH),
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    shuffle=True
)

test_data = test_gen.flow_from_directory(
    directory=str(TEST_PATH),
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

print(f"Class indices: {train_data.class_indices}")

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
output = Dense(len(CLASS_NAMES), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

model.save(str(MODEL_PATH))
print(f"✅ Model saved to: {MODEL_PATH}")
'''

predict_code = '''import argparse
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = Path("E:/Handloomproject/handloom_model.h5")
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
'''

Path('split.py').write_text(split_code, encoding='utf-8')
Path('HandloomGCN/dataset/train.py').write_text(train_code, encoding='utf-8')
Path('predict.py').write_text(predict_code, encoding='utf-8')
print('Updated split.py, HandloomGCN/dataset/train.py, and created predict.py')
