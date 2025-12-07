import sys
import os
import cv2
import numpy as np
import joblib

import features
from features import extract_cam_id, extract_features

# =======================================
# LOAD MODEL + METADATA
# =======================================
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found: {MODEL_PATH}")
    sys.exit(1)

DATA = joblib.load(MODEL_PATH)

model = DATA["model"]
label_encoder = DATA["label_encoder"]
scaler = DATA["scaler"]
feature_cols = DATA["feature_cols"]

print("[INFO] Model loaded.")
print("[INFO] Features expected:", feature_cols)

# =======================================
# PREDICT FUNCTION
# =======================================
def predict(path):
    img = cv2.imread(path)

    if img is None:
        print("[ERROR] Could not read image:", path)
        return None

    cam_id = extract_cam_id(path)
    if cam_id is None:
        print("[ERROR] Could not extract cam_id from filename:", path)
        print("File must start with camXX_... e.g. cam09_xxx.jpg")
        return None

    features.CURRENT_FILENAME = path
    feats = extract_features(img, cam_id)

    try:
        X = np.array([feats[col] for col in feature_cols]).reshape(1, -1)
    except KeyError as e:
        print(f"[ERROR] Missing feature for prediction: {e}")
        print("Extracted feats keys:", feats.keys())
        return None

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    label = label_encoder.inverse_transform(pred)[0]

    print("\n=== TRAFFIC PREDICTION RESULT ===")
    print(f"Image: {path}")
    print(f"Camera: {cam_id}")
    print(f"Predicted label: **{label.upper()}**")
    print("\nExtracted feature values:")
    for k in feature_cols:
        print(f"{k:20s}: {feats[k]}")

    print("\nDone.\n")
    return label

# =======================================
# MAIN
# =======================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter image path: ").strip()

    predict(img_path)
