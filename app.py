import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, request, url_for
import joblib
from werkzeug.utils import secure_filename

# =======================================
# CONFIG
# =======================================
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MODELS_PATH = os.path.join(BASE_PATH, "models")

UPLOAD_PATH = os.path.join(BASE_PATH, "static", "uploads")
os.makedirs(UPLOAD_PATH, exist_ok=True)

PIPELINE_A_PATH = os.path.join(BASE_PATH, "pipeline_a")
if PIPELINE_A_PATH not in sys.path:
    sys.path.insert(0, PIPELINE_A_PATH)

PIPELINE_B_PATH = os.path.join(BASE_PATH, "pipeline_b")
if PIPELINE_B_PATH not in sys.path:
    sys.path.insert(0, PIPELINE_B_PATH)

import extract_features
from extract_features import extract_features, extract_cam_id, is_non_traffic
from extract_deep_features import extract_deep_features

# =======================================
# AVAILABLE MODELS
# =======================================
MODEL_REGISTRY = {
    "hc":  {"file": "hc.pkl",  "label": "Handcrafted Feature", "pipeline": "A"},
    "rf":  {"file": "rf.pkl",  "label": "Random Forest", "pipeline": "B"},
    "gb":  {"file": "gb.pkl",  "label": "Gradient Boosting", "pipeline": "B"},
    "ada": {"file": "ada.pkl", "label": "AdaBoost", "pipeline": "B"},
    "xgb": {"file": "xgb.pkl", "label": "XGBoost", "pipeline": "B"},
    "svm": {"file": "svm.pkl", "label": "SVM", "pipeline": "B"},
}

# =======================================
# LOAD ALL MODELS
# =======================================
MODELS = {}

for key, cfg in MODEL_REGISTRY.items():
    model_path = os.path.join(MODELS_PATH, cfg["file"])
    if not os.path.exists(model_path):
        continue

    data = joblib.load(model_path)
    MODELS[key] = {
        "model": data["model"],
        "label_encoder": data["label_encoder"],
        "scaler": data["scaler"],
        "feature_cols": data["feature_cols"],
        "label": cfg["label"],
        "pipeline": cfg["pipeline"]
    }

print("[INFO] Loaded models:", list(MODELS.keys()))

# =======================================
# FLASK APP
# =======================================
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_PATH, "templates"),
    static_folder=os.path.join(BASE_PATH, "static")
)
app.config["UPLOAD_FOLDER"] = UPLOAD_PATH

# =======================================
# PREDICT FUNCTION
# =======================================
def predict(path, model_key):
    if model_key not in MODELS:
        raise ValueError("Invalid model selected")

    entry = MODELS[model_key]
    model = entry["model"]
    scaler = entry["scaler"]
    feature_cols = entry["feature_cols"]
    label_encoder = entry["label_encoder"]
    pipeline = entry["pipeline"]

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    if pipeline == "A":
        # extract features
        cam_id = extract_cam_id(path) or "default"
        extract_features.CURRENT_FILENAME = path
        feats = extract_features(img, cam_id)

        if is_non_traffic(feats):
            raise ValueError("Image does not appear to contain traffic")

        # prepare X in correct order
        X = np.array([feats[col] for col in feature_cols]).reshape(1, -1)
    else:
        deep_feat = extract_deep_features(path)
        X = deep_feat.reshape(1, -1)
        feats = None

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    label = label_encoder.inverse_transform(pred)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)
        confidence = float(np.max(prob))

    return {
        "label": label,
        "features": feats,
        "pipeline": pipeline,
        "confidence": confidence
    }

# =======================================
# ROUTES
# =======================================
@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    selected_models = request.form.getlist("model")
    if not selected_models:
        selected_models = ["hc"]

    if request.method == "POST":
        files = request.files.getlist("images")

        for file in files:
            if file.filename.strip() == "":
                continue

            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_PATH, filename)
            file.save(save_path)

            rel_path = os.path.relpath(save_path, app.static_folder)
            public_url = url_for("static", filename=rel_path)
            predictions = []

            for model_key in selected_models:
                r = predict(save_path, model_key)
                predictions.append({
                    "model_key": model_key,
                    "model_label": MODEL_REGISTRY[model_key]["label"],
                    "pipeline": r["pipeline"],
                    "label": r["label"],
                    "confidence": r["confidence"],
                    "features": r["features"],
                })

            results.append({
                "filename": filename,
                "path": public_url,
                "predictions": predictions
            })

    return render_template(
        "index.html",
        results=results,
        models=MODEL_REGISTRY
    )

# =======================================
# MAIN
# =======================================
if __name__ == "__main__":
    app.run(debug=True)
