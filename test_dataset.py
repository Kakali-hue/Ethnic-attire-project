import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import defaultdict

# Paths
dataset_root = "E:/Handloomproject/HandloomGCN/dataset"
train_path = os.path.join(dataset_root, "train")
test_path = os.path.join(dataset_root, "test")
model_path = "E:/Handloomproject/handloom_model.h5"

# Classes
classes = ["muga", "nuni", "pat"]

def check_dataset_structure():
    """Check dataset folder structure and image counts"""
    print("=" * 60)
    print("📊 DATASET STRUCTURE CHECK")
    print("=" * 60)
    
    stats = {}
    
    for split, path in [("Train", train_path), ("Test", test_path)]:
        print(f"\n{split} Dataset:")
        print("-" * 40)
        split_stats = {}
        
        if not os.path.exists(path):
            print(f"❌ {split} path does not exist: {path}")
            continue
            
        total_images = 0
        for class_name in classes:
            class_path = os.path.join(path, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
                count = len(images)
                split_stats[class_name] = count
                total_images += count
                print(f"  {class_name.upper():10} - {count:3d} images")
            else:
                print(f"  ❌ Missing folder: {class_name}")
                split_stats[class_name] = 0
        
        print(f"  {'TOTAL':10} - {total_images:3d} images")
        stats[split] = split_stats
    
    return stats

def check_image_validity():
    """Check if all images can be loaded properly"""
    print("\n" + "=" * 60)
    print("🖼️  IMAGE VALIDITY CHECK")
    print("=" * 60)
    
    invalid_images = []
    
    for split, path in [("Train", train_path), ("Test", test_path)]:
        print(f"\n{split} Dataset:")
        print("-" * 40)
        
        for class_name in classes:
            class_path = os.path.join(path, class_name)
            if not os.path.exists(class_path):
                continue
                
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
            invalid_count = 0
            
            for img_file in images:
                img_path = os.path.join(class_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        invalid_count += 1
                        invalid_images.append((split, class_name, img_file))
                except Exception as e:
                    invalid_count += 1
                    invalid_images.append((split, class_name, img_file))
            
            status = "✅" if invalid_count == 0 else "⚠️"
            print(f"  {class_name.upper():10} - {invalid_count} invalid images")
    
    if invalid_images:
        print("\n⚠️  Invalid images found:")
        for split, class_name, img_file in invalid_images:
            print(f"  {split}/{class_name}/{img_file}")
    else:
        print("\n✅ All images are valid!")
    
    return invalid_images

def test_model_on_dataset():
    """Test the model on the test dataset"""
    print("\n" + "=" * 60)
    print("🤖 MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    try:
        model = load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    class_results = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0
    overall_total = 0
    
    print("\nTesting on test dataset...")
    print("-" * 40)
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(test_path, class_name)
        if not os.path.exists(class_path):
            continue
            
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        for img_file in images:
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Preprocess image
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                img = np.reshape(img, (1, 224, 224, 3))
                
                # Predict
                prediction = model.predict(img, verbose=0)
                predicted_class_idx = np.argmax(prediction)
                predicted_class = classes[predicted_class_idx]
                
                class_results[class_name]["total"] += 1
                overall_total += 1
                
                if predicted_class == class_name:
                    class_results[class_name]["correct"] += 1
                    overall_correct += 1
            except Exception as e:
                pass
    
    # Print results
    print("\nResults by class:")
    print("-" * 40)
    for class_name in classes:
        if class_results[class_name]["total"] > 0:
            accuracy = (class_results[class_name]["correct"] / class_results[class_name]["total"]) * 100
            print(f"  {class_name.upper():10} - {accuracy:.2f}% ({class_results[class_name]['correct']}/{class_results[class_name]['total']})")
    
    if overall_total > 0:
        overall_accuracy = (overall_correct / overall_total) * 100
        print("\n" + "=" * 40)
        print(f"  OVERALL   - {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")
        print("=" * 40)
    else:
        print("❌ No valid images found to test!")

def main():
    """Run all tests"""
    print("\n")
    print("🧪 HANDLOOM DATASET TEST SUITE")
    print("=" * 60)
    print(f"Dataset location: {dataset_root}")
    print(f"Model location: {model_path}")
    print()
    
    # Run tests
    check_dataset_structure()
    invalid_images = check_image_validity()
    test_model_on_dataset()
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
