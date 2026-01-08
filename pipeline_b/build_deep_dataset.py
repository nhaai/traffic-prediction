import os
import pandas as pd
from tqdm import tqdm

from extract_deep_features import extract_deep_features

# =======================================
# CONFIG
# =======================================
CLEAN_DIR = "dataset_cleaned"
OUTPUT_CSV = "pipeline_b/dataset_deep_features.csv"
LABELS = ["free_flow", "moderate", "congested"]

# =======================================
# BUILD DATASET
# =======================================
def build_dataset():
    rows = []
    for label in LABELS:
        label_dir = os.path.join(CLEAN_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        for f in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(label_dir, f)
            feats = extract_deep_features(path)

            row = {f"f{i}": feats[i] for i in range(len(feats))}
            row["label"] = label
            row["cam_id"] = "default"
            row["filepath"] = path
            rows.append(row)

    # write CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] Dataset built with {len(df)} samples")
    return df

# =======================================
# MAIN
# =======================================
if __name__ == "__main__":
    build_dataset()
