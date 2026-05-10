import random
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
