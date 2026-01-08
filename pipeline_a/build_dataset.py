import argparse
import os
import shutil
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import extract_features
from extract_features import extract_features, extract_cam_id

IMG_SIZE = 640
RAW_DIR = "dataset_raw"
CLEAN_DIR = "dataset_cleaned"
OUTPUT_CSV = "pipeline_a/dataset_features.csv"
LABELS = ["free_flow", "moderate", "congested"]

# =======================================
# ARGS
# =======================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--refresh",
    type=lambda x: x.lower() == "true",
    default=True,
    help="Rebuild cleaned dataset and resized images"
)
args = parser.parse_args()

if args.refresh:
    shutil.rmtree(CLEAN_DIR, ignore_errors=True)
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
os.makedirs(CLEAN_DIR, exist_ok=True)

# =======================================
# PROCESS DATASET
# =======================================
def build_dataset():
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

        # resize for training
        clean_label_dir = os.path.join(CLEAN_DIR, label)
        os.makedirs(clean_label_dir, exist_ok=True)
        clean_path = os.path.join(clean_label_dir, filename)

        if args.refresh or not os.path.exists(clean_path):
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(clean_path, img_resized)
        else:
            img_resized = cv2.imread(clean_path)
            if img_resized is None:
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(clean_path, img_resized)

        # extract features
        cam_id = extract_cam_id(filename)
        if cam_id is None:
            cam_id = "default"

        extract_features.CURRENT_FILENAME = clean_path
        feats = extract_features(img_resized, cam_id)
        feats["label"] = label
        feats["cam_id"] = cam_id
        feats["filepath"] = clean_path

        rows.append(feats)
        pbar.update(1)
    pbar.close()

    # write CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] Dataset built with {len(df)} samples")
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

    train_df.to_csv("pipeline_a/train.csv", index=False)
    val_df.to_csv("pipeline_a/val.csv", index=False)
    test_df.to_csv("pipeline_a/test.csv", index=False)

    print("[DONE] Saved train.csv / val.csv / test.csv")

# =======================================
# MAIN
# =======================================
if __name__ == "__main__":
    processed = build_dataset()
    split_dataset(processed)
