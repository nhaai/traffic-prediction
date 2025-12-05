import sys
import cv2
import numpy as np
import joblib
from ultralytics import YOLO

# =========================================================
# LOAD TRAINED MODEL + YOLOv8
# =========================================================
DATA = joblib.load("model.pkl")

model = DATA["model"]
label_encoder = DATA["label_encoder"]
scaler = DATA["scaler"]
feature_cols = DATA["feature_cols"]

yolo = YOLO("yolov8s.pt")
IMG_SIZE = 640

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

    return feats, ("NIGHT" if feats["is_night"] == 1 else "DAY")

# =========================================================
# PREDICT FUNCTION
# =========================================================
def predict(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("[ERROR] Could not read image:", img_path)
        return

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    feats, mode = extract_features(img_resized)

    X = np.array([feats[col] for col in feature_cols]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    label = label_encoder.inverse_transform(pred)[0]

    print("\n=== TRAFFIC PREDICTION RESULT ===")
    print(f"Image: {img_path}")
    print(f"Mode: {mode}")
    print(f"Predicted label: **{label.upper()}**")
    print("\nExtracted feature values:")
    for k, v in feats.items():
        print(f"{k:20s}: {v}")

    print("\nDone.\n")
    return label

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter image path: ").strip()
    predict(path)
