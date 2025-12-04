import os
import cv2
import numpy as np
from flask import Flask, render_template, request
import joblib
from ultralytics import YOLO
from demo_predict import extract_features
from werkzeug.utils import secure_filename

# ===================================================
# CONFIG
# ===================================================
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMG_SIZE = 640

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

# ===================================================
# LOAD MODEL + YOLO
# ===================================================
data = joblib.load("model.pkl")
model = data["model"]
label_encoder = data["label_encoder"]
scaler = data["scaler"]
feature_cols = data["feature_cols"]

yolo = YOLO("yolov8s.pt")

# ===================================================
# PREDICT FUNCTION (reuse logic)
# ===================================================
def predict_image(path):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    feats = extract_features(img_resized)

    X = np.array([feats[col] for col in feature_cols]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    label = label_encoder.inverse_transform(pred)[0]

    return label, feats

# ===================================================
# ROUTES
# ===================================================
@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        files = request.files.getlist("images")

        for file in files:
            if file.filename == "":
                continue

            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_DIR, filename)
            file.save(save_path)

            label, feats = predict_image(save_path)

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
