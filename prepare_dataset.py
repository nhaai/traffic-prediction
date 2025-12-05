import os
import cv2
import numpy as np
import shutil
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from tqdm import tqdm

RAW_DIR = "dataset_raw"
CLEAN_DIR = "dataset_cleaned"
SPLIT_DIR = "dataset_split"
OUTPUT_CSV = "dataset_features.csv"
IMG_SIZE = 640

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# =========================================================
# CLI ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser(description="Dataset processor")
parser.add_argument(
    "--reset",
    action="store_true",
    help="Reset dataset_cleaned, dataset_split and dataset_features.csv before processing"
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Auto-confirm reset without interactive prompt (useful for non-interactive shells)"
)
args = parser.parse_args()

# =========================================================
# RESET DATASET
# =========================================================
def reset_dataset(auto_confirm=False):
    print("WARNING: You are about to delete all processed datasets:")
    print("- dataset_cleaned/")
    print("- dataset_split/")
    print("- dataset_features.csv")
    print("Raw images in dataset_raw/ will NOT be touched.")

    if auto_confirm:
        confirm = "YES"
    else:
        try:
            confirm = input("Type 'YES' to confirm: ")
        except (UnicodeDecodeError, Exception) as e:
            print("Warning: interactive input failed:", str(e))
            print("Use --yes to force reset in non-interactive environments.")
            return False

    if confirm == "YES":
        print("Deleting old processed datasets...")
        shutil.rmtree(CLEAN_DIR, ignore_errors=True)
        shutil.rmtree(SPLIT_DIR, ignore_errors=True)
        if os.path.exists(OUTPUT_CSV):
            os.remove(OUTPUT_CSV)

        os.makedirs(CLEAN_DIR, exist_ok=True)
        os.makedirs(SPLIT_DIR, exist_ok=True)

        print("Reset completed.")
        return True
    else:
        print("Reset cancelled. Exit.")
        return False

if args.reset:
    ok = reset_dataset(auto_confirm=args.force)
    if not ok:
        print("Exiting due to cancelled reset.")
        exit(1)

# =========================================================
# LOAD MODEL
# =========================================================
def load_model():
    import torch
    from packaging import version

    weights_path = "yolov8s.pt"

    # Auto download if missing
    if not os.path.exists(weights_path):
        YOLO("yolov8s.pt")

    torch_version = torch.__version__

    # For PyTorch >= 2.6, must use safe_globals workaround
    if version.parse(torch_version) >= version.parse("2.6.0"):
        print("PyTorch >= 2.6 detected — enabling safe global loader to avoid pickle errors.")

        try:
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
            print("Safe globals added.")
        except Exception as e:
            print("Failed to add safe globals:", e)
            print("Trying fallback loader...")

    # Try loading the model normally
    try:
        yolo_model = YOLO(weights_path)
        return yolo_model
    except Exception as e:
        print("Normal load failed:", type(e).__name__, str(e))
        print("Attempting fallback torch.load() with weights_only=False...")

        # Fallback load (allowed only for trusted model files)
        try:
            ckpt = torch.load(weights_path, weights_only=False, map_location="cpu")
            print("Checkpoint loaded manually, constructing model...")

            from ultralytics.nn.tasks import DetectionModel

            model = DetectionModel()
            model.load_state_dict(ckpt["model"].state_dict())

            return YOLO(model=model)
        except Exception as e2:
            print("Fallback load also failed:", type(e2).__name__, str(e2))
            print("Please check your PyTorch / Ultralytics installation.")
            raise e2

yolo = load_model()

# =========================================================
# EXTRACT FEATURES
# =========================================================
def is_night(gray):
    brightness = np.mean(gray)
    return brightness < 90

def apply_night_adjustments(feats, gray):
    if is_night(gray):
        feats["car_count"] *= 0.6
        feats["motorcycle_count"] *= 0.5
        feats["bus_count"] *= 0.8
        feats["truck_count"] *= 0.8
        feats["total_vehicles"] *= 0.6
        feats["bbox_area_ratio"] *= 1.4
        feats["mean_bbox_area"] *= 1.2
        feats["max_bbox_area"] *= 1.1
        feats["edge_density"] *= 0.7
        feats["sharpness"] *= 1.1
        feats["is_night"] = 1
    else:
        feats["is_night"] = 0

def extract_features(img):
    h, w, _ = img.shape
    area = h * w

    # Run detection
    results = yolo(img, verbose=False)[0]

    # Detect specific vehicle classes from COCO dataset
    vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    counts = {v: 0 for v in vehicle_classes.values()}
    bbox_areas = []

    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls in vehicle_classes and conf >= 0.15: # lower conf for night
            label = vehicle_classes[cls]
            counts[label] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_areas.append((x2 - x1) * (y2 - y1))

    total_vehicles = sum(counts.values())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = {
        "car_count": counts["car"],
        "motorcycle_count": counts["motorcycle"],
        "bus_count": counts["bus"],
        "truck_count": counts["truck"],
        "total_vehicles": total_vehicles,
        "bbox_area_ratio": sum(bbox_areas) / area if bbox_areas else 0,
        "mean_bbox_area": float(np.mean(bbox_areas)) if bbox_areas else 0,
        "max_bbox_area": max(bbox_areas) if bbox_areas else 0,
        "brightness": float(np.mean(gray)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float(np.sum(cv2.Canny(gray, 80, 160) > 0) / area),
    }
    apply_night_adjustments(feats, gray)

    return feats

# =========================================================
# LABELING
# =========================================================
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

# =========================================================
# PROCESS DATASET
# =========================================================
def process_dataset():
    # Load old CSV
    if os.path.exists(OUTPUT_CSV):
        old_df = pd.read_csv(OUTPUT_CSV)
        processed_files = set(old_df["filepath"].apply(lambda x: os.path.basename(x)))
        print(f"Loaded {len(old_df)} existing samples.")
    else:
        old_df = pd.DataFrame()
        processed_files = set()
        print("No existing CSV found. Starting fresh.")

    # List all raw images
    raw_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    # Find new files
    new_files = [f for f in raw_files if f not in processed_files]
    print(f"Total raw files: {len(raw_files)}")
    print(f"New files to process: {len(new_files)}")

    rows = []
    total_steps = len(new_files) + 1
    pbar = tqdm(total=total_steps, desc="Processing images", unit="img")

    try:
        for filename in new_files:
            raw_path = os.path.join(RAW_DIR, filename)
            img = cv2.imread(raw_path)
            if img is None:
                print("Cannot read:", raw_path)
                pbar.update(1)
                continue

            # Resize for consistent input
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Save cleaned image
            clean_path = os.path.join(CLEAN_DIR, filename)
            if not os.path.exists(clean_path):
                cv2.imwrite(clean_path, img_resized)

            # Extract features
            f = extract_features(img_resized)
            f["filepath"] = clean_path
            f["label"] = auto_label(f)
            rows.append(f)
            pbar.update(1)
        pbar.update(1)
    finally:
        pbar.close()

    # Build new DF
    new_df = pd.DataFrame(rows)

    # Append to old
    final_df = pd.concat([old_df, new_df], ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] Dataset now contains {len(final_df)} samples.")
    return final_df

# =========================================================
# SPLIT DATASET
# =========================================================
def split_dataset(df):
    # Clean old split folders
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR, ignore_errors=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42, stratify=temp_df["label"])

    # Save CSVs
    train_df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, "test.csv"), index=False)

    # Copy images
    def copy_files(split_df, split_name):
        for _, row in split_df.iterrows():
            lbl = row["label"]
            dst_dir = os.path.join(SPLIT_DIR, split_name, lbl)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(row["filepath"], dst_dir)

    copy_files(train_df, "train")
    copy_files(val_df, "val")
    copy_files(test_df, "test")

    print("[DONE] Rebuilt train/val/test split.")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    processed = process_dataset()
    split_dataset(processed)
    print("All completed.")
