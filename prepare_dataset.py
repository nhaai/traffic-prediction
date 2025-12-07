import os
import cv2
import shutil
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import features
from features import extract_features, extract_cam_id
from labeling import auto_label

RAW_DIR = "dataset_raw"
CLEAN_DIR = "dataset_cleaned"
SPLIT_DIR = "dataset_split"
OUTPUT_CSV = "dataset_features.csv"
IMG_SIZE = 640

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# =======================================
# CLI
# =======================================
parser = argparse.ArgumentParser(description="Dataset processor")
parser.add_argument(
    "--reset",
    action="store_true",
    help="Reset dataset_cleaned, dataset_split and dataset_features.csv before processing"
)
args = parser.parse_args()

# =======================================
# RESET DATASET
# =======================================
def reset_dataset(auto_confirm=False):
    print("WARNING: This will delete:")
    print("- dataset_cleaned/")
    print("- dataset_split/")
    print("- dataset_features.csv")
    print("Raw images in dataset_raw/ are safe.")

    if auto_confirm:
        confirm = "YES"
    else:
        try:
            confirm = input("Type 'YES' to confirm: ")
        except (UnicodeDecodeError, Exception) as e:
            print("Cannot read input.")
            return False

    if confirm == "YES":
        print("Deleting old datasets...")
        shutil.rmtree(CLEAN_DIR, ignore_errors=True)
        shutil.rmtree(SPLIT_DIR, ignore_errors=True)
        if os.path.exists(OUTPUT_CSV):
            os.remove(OUTPUT_CSV)

        os.makedirs(CLEAN_DIR, exist_ok=True)
        os.makedirs(SPLIT_DIR, exist_ok=True)

        print("Reset completed.")
        return True

    print("Cancelled.")
    return False

if args.reset:
    ok = reset_dataset(auto_confirm=True)
    if not ok:
        exit(1)

# =======================================
# PROCESS DATASET
# =======================================
def process_dataset():
    # Load old CSV
    if os.path.exists(OUTPUT_CSV):
        old_df = pd.read_csv(OUTPUT_CSV)
        processed_files = set(old_df["filepath"].apply(lambda x: os.path.basename(x)))
        print(f"Loaded {len(old_df)} existing samples.")
    else:
        old_df = pd.DataFrame()
        processed_files = set()
        print("No existing CSV. Starting fresh.")

    # List all raw images
    raw_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    new_files = [f for f in raw_files if f not in processed_files]
    print(f"Total raw images: {len(raw_files)}")
    print(f"New images to process: {len(new_files)}")

    rows = []
    pbar = tqdm(total=len(new_files), desc="Processing images", unit="img")

    for filename in new_files:
        raw_path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(raw_path)
        if img is None:
            print("Cannot read:", raw_path)
            pbar.update(1)
            continue

        cam_id = extract_cam_id(filename)
        if cam_id is None:
            print("Cannot extract cam_id from filename:", filename)
            pbar.update(1)
            continue

        # Resize for training
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        clean_path = os.path.join(CLEAN_DIR, filename)
        if not os.path.exists(clean_path):
            cv2.imwrite(clean_path, img_resized)

        # Extract features
        features.CURRENT_FILENAME = clean_path
        feats = extract_features(img, cam_id)
        feats["filepath"] = clean_path
        feats["cam_id"] = cam_id
        feats["label"] = auto_label(feats, cam_id)

        rows.append(feats)
        pbar.update(1)

    pbar.close()

    # Merge CSV
    new_df = pd.DataFrame(rows)
    final_df = pd.concat([old_df, new_df], ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] Dataset now contains {len(final_df)} samples.")
    return final_df

# =======================================
# SPLIT DATASET
# =======================================
def split_dataset(df):
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

    print("[DONE] Split train/val/test.")

# =======================================
# MAIN
# =======================================
if __name__ == "__main__":
    processed = process_dataset()
    split_dataset(processed)
    print("All completed.")
