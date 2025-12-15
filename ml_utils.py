import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.pkl"
CSV_PATH = "dataset_features.csv"

# =======================================
# Load model.pkl
# =======================================
def load_model(model_path=MODEL_PATH):
    """
    Load model.pkl and return:
    model, label_encoder, scaler, feature_cols, best_params
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    data = joblib.load(model_path)

    required = ["model", "label_encoder", "scaler", "feature_cols"]
    for key in required:
        if key not in data:
            raise KeyError(f"model.pkl missing key: {key}")

    return (
        data["model"],
        data["label_encoder"],
        data["scaler"],
        data["feature_cols"],
        data.get("best_params", None)
    )

# =======================================
# Load dataset & extract features
# =======================================
def load_dataset(csv_path=CSV_PATH, feature_cols=None, label_encoder=None):
    """
    Load CSV into df, and return:
        df, X (numpy), y_raw (labels string), y (encoded)
    If feature_cols is provided, only use those columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ["label", "cam_id", "filepath"]]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print("\n[WARNING] Missing feature columns in CSV:")
        for m in missing:
            print(" -", m)
        print("The model may behave unpredictably.\n")

    used_cols = [c for c in feature_cols if c in df.columns]
    X = df[used_cols].values
    y_raw = df["label"].values

    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = label_encoder.transform(y_raw)

    return df, X, y_raw, y, used_cols, label_encoder

def load_split_dataset(split="train", feature_cols=None, label_encoder=None):
    csv_map = {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",
    }
    if split not in csv_map:
        raise ValueError("split must be train / val / test")

    return load_dataset(csv_map[split], feature_cols=feature_cols, label_encoder=label_encoder)

# =======================================
# Fit scaling for training
# =======================================
def fit_scaler(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled

# =======================================
# Transform using existing scaler
# =======================================
def transform_with_scaler(X, scaler):
    return scaler.transform(X)

# =======================================
# Save model.pkl
# =======================================
def save_model(model_path, model, label_encoder, scaler, feature_cols, best_params=None):
    joblib.dump({
        "model": model,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "best_params": best_params
    }, model_path)

    print(f"\n[SAVED] Model saved to {model_path}")

# =======================================
# Predict using model + scaler
# =======================================
def predict_with_model(model, scaler, X):
    """
    Apply scaler.transform → model.predict
    """
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)
