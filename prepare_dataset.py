import os
import shutil
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import features
from features import extract_features, extract_cam_id

IMG_SIZE = 640
LABELS = ["free_flow", "moderate", "congested"]
OUTPUT_CSV = "dataset_features.csv"
CLEAN_DIR = "dataset_cleaned"
RAW_DIR = "dataset_raw"

# =======================================
# PREPARE OUTPUT DIRS
# =======================================
shutil.rmtree(CLEAN_DIR, ignore_errors=True)
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)
os.makedirs(CLEAN_DIR, exist_ok=True)

# =======================================
# PROCESS DATASET
# =======================================
def process_dataset():
    raw_files = []
    for label in LABELS:
        label_dir = os.path.join(RAW_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for f in os.listdir(label_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                raw_files.append((label, f))
    print(f"Total raw images to process: {len(raw_files)}")

    rows = []
    pbar = tqdm(total=len(raw_files), desc="Processing images", unit="img")
    for label, filename in raw_files:
        raw_path = os.path.join(RAW_DIR, label, filename)
        img = cv2.imread(raw_path)
        if img is None:
            print("Cannot read:", raw_path)
            pbar.update(1)
            continue

        cam_id = extract_cam_id(filename)
        if cam_id is None:
            cam_id = "default"

        # resize for training
        clean_label_dir = os.path.join(CLEAN_DIR, label)
        os.makedirs(clean_label_dir, exist_ok=True)

        clean_path = os.path.join(clean_label_dir, filename)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(clean_path, img_resized)

        # extract features
        features.CURRENT_FILENAME = clean_path
        feats = extract_features(img_resized, cam_id)
        feats["filepath"] = clean_path
        feats["cam_id"] = cam_id
        feats["label"] = label

        rows.append(feats)
        pbar.update(1)
    pbar.close()

    # write CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] Dataset built with {len(df)} samples.")
    return df

# =======================================
# SPLIT DATASET
# =======================================
def split_dataset(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=1/3, random_state=42, stratify=temp_df["label"]
    )

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print("[DONE] Saved train.csv / val.csv / test.csv")

# =======================================
# MAIN
# =======================================
if __name__ == "__main__":
    processed = process_dataset()
    split_dataset(processed)
