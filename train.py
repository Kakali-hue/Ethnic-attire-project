from pathlib import Path
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
