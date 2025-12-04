import os
import cv2
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

RAW_DIR = "dataset_raw"
CLEAN_DIR = "dataset_cleaned"
SPLIT_DIR = "dataset_split"
IMG_SIZE = 640
OUTPUT_CSV = "dataset_features.csv"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# ============================================================
# LOAD OR DOWNLOAD MODEL
# ============================================================
def load_model():
    if not os.path.exists("yolov8s.pt"):
        print("Downloading YOLOv8s model...")
        model = YOLO("yolov8s.pt")
    else:
        model = YOLO("yolov8s.pt")
    print("YOLOv8s loaded.")
    return model

model = load_model()

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(img):
    h, w, _ = img.shape
    area = h * w

    # Run detection
    results = model(img, verbose=False)[0]

    # Detect specific vehicle classes from COCO dataset
    vehicle_classes = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    bbox_areas = []
    counts = {"car":0, "motorcycle":0, "bus":0, "truck":0}

    for box in results.boxes:
        cls = int(box.cls)
        if cls in vehicle_classes:
            label = vehicle_classes[cls]
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_area = (x2 - x1) * (y2 - y1)
            bbox_areas.append(bbox_area)

    total_vehicles = sum(counts.values())
    bbox_area_ratio = sum(bbox_areas) / area if bbox_areas else 0
    mean_bbox_area = np.mean(bbox_areas) if bbox_areas else 0
    max_bbox_area = max(bbox_areas) if bbox_areas else 0

    # Additional features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.sum(edges > 0) / area)

    return {
        "car_count": counts["car"],
        "motorcycle_count": counts["motorcycle"],
        "bus_count": counts["bus"],
        "truck_count": counts["truck"],
        "total_vehicles": total_vehicles,
        "bbox_area_ratio": bbox_area_ratio,
        "mean_bbox_area": mean_bbox_area,
        "max_bbox_area": max_bbox_area,
        "brightness": brightness,
        "sharpness": sharpness,
        "edge_density": edge_density,
    }

# ============================================================
# AUTO LABELING RULE (BASED ON REAL DETECTION)
# ============================================================
def auto_label(f):
    total = f["total_vehicles"]
    dens = f["bbox_area_ratio"]

    # Adjust thresholds depending on camera perspective
    if total <= 5 and dens < 0.05:
        return "free_flow"
    elif total <= 15 and dens < 0.15:
        return "moderate"
    else:
        return "congested"

# ============================================================
# PROCESS DATASET
# ============================================================
def process_dataset():
    rows = []
    file_list = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Found {len(file_list)} raw images.")

    for filename in file_list:
        path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(path)

        if img is None:
            print("Warning: cannot read image:", path)
            continue

        # Resize for consistent input
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Save cleaned image
        clean_path = os.path.join(CLEAN_DIR, filename)
        cv2.imwrite(clean_path, img_resized)

        # Extract features
        f = extract_features(img_resized)
        f["filepath"] = clean_path
        f["label"] = auto_label(f)
        rows.append(f)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Created {OUTPUT_CSV} with {len(df)} samples.")
    return df

# ============================================================
# SPLIT DATASET
# ============================================================
def split_dataset(df):
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42, stratify=temp_df["label"])

    train_df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, "test.csv"), index=False)

    def copy_split(split_df, split_name):
        for _, row in split_df.iterrows():
            lbl = row["label"]
            dst_dir = os.path.join(SPLIT_DIR, split_name, lbl)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(row["filepath"], dst_dir)

    copy_split(train_df, "train")
    copy_split(val_df, "val")
    copy_split(test_df, "test")

    print("[DONE] Dataset split into train/val/test.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = process_dataset()
    split_dataset(df)
    print("All completed.")
