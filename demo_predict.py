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
# EXTRACT FEATURES (SAME AS TRAINING)
# =========================================================
def extract_features(img):
    h, w, _ = img.shape
    area = h * w

    results = yolo(img, verbose=False)[0]

    vehicle_classes = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    counts = {"car":0, "motorcycle":0, "bus":0, "truck":0}
    bbox_areas = []

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

# =========================================================
# PREDICT FUNCTION
# =========================================================
def predict(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("[ERROR] Could not read image:", img_path)
        return

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    feats = extract_features(img_resized)

    X = np.array([feats[col] for col in feature_cols]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    label = label_encoder.inverse_transform(pred)[0]

    print("\n=== TRAFFIC PREDICTION RESULT ===")
    print(f"Image: {img_path}")
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
    path = input("Enter image path: ").strip()
    predict(path)
