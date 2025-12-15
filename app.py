import os
import cv2
import numpy as np
from flask import Flask, render_template, request
import joblib
from werkzeug.utils import secure_filename

import features
from features import extract_features, extract_cam_id

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

# ===================================================
# LOAD MODEL + METADATA
# ===================================================
data = joblib.load("model.pkl")
model = data["model"]
label_encoder = data["label_encoder"]
scaler = data["scaler"]
feature_cols = data["feature_cols"]

print("[INFO] Model loaded.")
print("[INFO] Required features:", feature_cols)

# ===================================================
# PREDICT FUNCTION
# ===================================================
def predict(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    cam_id = extract_cam_id(path)
    if cam_id is None:
        cam_id = "default"

    # extract features
    features.CURRENT_FILENAME = path
    feats = extract_features(img, cam_id)

    if features.is_non_traffic(feats):
        raise ValueError("Image does not appear to contain traffic.")

    # prepare X in correct order
    X = np.array([feats[col] for col in feature_cols]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    label = label_encoder.inverse_transform(pred)[0]

    return label, feats

def resize_safe(img, max_size=1280):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# ===================================================
# ROUTES
# ===================================================
@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        files = request.files.getlist("images")

        for file in files:
            if file.filename.strip() == "":
                continue

            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_DIR, filename)
            file.save(save_path)

            try:
                label, feats = predict(save_path)
            except Exception as e:
                results.append({
                    "filename": filename,
                    "path": save_path,
                    "label": f"ERROR: {e}",
                    "features": {}
                })
                continue

            results.append({
                "filename": filename,
                "path": save_path,
                "label": label,
                "features": feats
            })

    return render_template("index.html", results=results)

# ===================================================
# MAIN
# ===================================================
if __name__ == "__main__":
    app.run(debug=True)
