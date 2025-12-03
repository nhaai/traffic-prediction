import os
import cv2
import numpy as np
import shutil
import random
from sklearn.model_selection import train_test_split
import pandas as pd

# ============================================
# CONFIG
# ============================================
RAW_DATASET_DIR = "dataset_raw"            # contains 3 folders: free_flow / moderate / congested
CLEAN_DATASET_DIR = "dataset_cleaned"      # cleaned & resized images
SPLIT_DATASET_DIR = "dataset_split"        # train/val/test

IMG_SIZE = 224   # resize to 224×224

# ensure folders exist
os.makedirs(CLEAN_DATASET_DIR, exist_ok=True)
os.makedirs(SPLIT_DATASET_DIR, exist_ok=True)


# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_features(image):
    """
    Extract features from image:
    - vehicle_count: number of large detected objects (approx. number of vehicles)
    - density_ratio: ratio of white pixels (proxy for traffic density)
    - motion_intensity: optical flow magnitude (not applicable for single images → set to 0)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) Background subtraction → detect foreground objects
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=50)
    fgMask = backSub.apply(gray)
    _, thresh = cv2.threshold(fgMask, 127, 255, cv2.THRESH_BINARY)

    # 2) Find contours to estimate number of vehicles
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vehicle_cnt = sum(1 for c in contours if cv2.contourArea(c) > 300)   # count only large contours

    # 3) Density ratio = percentage of white pixels
    density_ratio = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])

    # 4) Optical flow (not usable for a single image → set to 0)
    motion_intensity = 0.0

    return vehicle_cnt, density_ratio, motion_intensity


# ============================================
# CLEAN + RESIZE + FEATURE EXTRACTION
# ============================================

def process_dataset():
    rows = []   # feature information → CSV

    for label in ["free_flow", "moderate", "congested"]:
        input_dir = os.path.join(RAW_DATASET_DIR, label)
        output_dir = os.path.join(CLEAN_DATASET_DIR, label)

        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print("Failed to read image:", img_path)
                continue

            # Simple image cleaning: mild Gaussian blur
            img = cv2.GaussianBlur(img, (3, 3), 0)

            # Resize
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Save cleaned image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_resized)

            # Extract features
            vehicle_cnt, density_ratio, motion_intensity = extract_features(img_resized)

            rows.append({
                "filepath": output_path,
                "vehicle_count": vehicle_cnt,
                "density_ratio": density_ratio,
                "motion_intensity": motion_intensity,
                "label": label,
            })

    # Save dataset for ML training
    df = pd.DataFrame(rows)
    df.to_csv("dataset_features.csv", index=False)
    print("Created dataset_features.csv with", len(df), "samples.")


# ============================================
# SPLIT DATASET 70/20/10
# ============================================

def split_dataset():
    df = pd.read_csv("dataset_features.csv")

    # Train 70%, temp 30%
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )

    # From the remaining 30% → val 20% and test 10%
    val_df, test_df = train_test_split(
        temp_df, test_size=1/3, random_state=42, stratify=temp_df["label"]
    )

    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    # Save CSV
    train_df.to_csv(os.path.join(SPLIT_DATASET_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_DATASET_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_DATASET_DIR, "test.csv"), index=False)

    # Copy images according to split
    def copy_images(df, split_name):
        out_dir = os.path.join(SPLIT_DATASET_DIR, split_name)
        os.makedirs(out_dir, exist_ok=True)

        for _, row in df.iterrows():
            label = row["label"]
            src = row["filepath"]
            dst_dir = os.path.join(out_dir, label)
            os.makedirs(dst_dir, exist_ok=True)

            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copy(src, dst)

    copy_images(train_df, "train")
    copy_images(val_df, "val")
    copy_images(test_df, "test")

    print("Dataset split into train/val/test.")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=== START CLEANING + FEATURE EXTRACTION ===")
    process_dataset()

    print("=== START DATASET SPLITTING ===")
    split_dataset()

    print("=== DONE ===")
