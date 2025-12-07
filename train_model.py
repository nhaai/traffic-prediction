import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
from tqdm import tqdm
import time
import os
import sys

CSV_PATH = "dataset_features.csv"
MODEL_PATH = "model.pkl"

# =======================================
# LOAD CSV
# =======================================
if not os.path.exists(CSV_PATH):
    print(f"CSV not found: {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print("Loaded CSV rows:", len(df))
print(df.head())

expected_feature_cols = [
    "car",
    "motorcycle",
    "bus",
    "truck",
    "total",

    "bbox_area_ratio",
    "mean_bbox_area",
    "max_bbox_area",
    "brightness",
    "sharpness",
    "edge_density",

    "zone_top",
    "zone_mid",
    "zone_bottom",

    "bottom_motor",
    "mid_car",
    "cluster_density",

    "is_night",
    "is_rain",
    "crowd_density"
]
feature_cols = [c for c in expected_feature_cols if c in df.columns]

missing = set(expected_feature_cols) - set(feature_cols)
if missing:
    print("\n[WARNING] Missing feature columns in CSV:")
    for m in missing:
        print(" -", m)
    print("Training will proceed using available features only.\n")

print("\nUsing feature columns:", feature_cols, "\n")

X = df[feature_cols].values
y_raw = df["label"].values

# =======================================
# LABEL ENCODER
# =======================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# =======================================
# NORMALIZATION
# (Decision Tree does not require it, but OK for consistency)
# =======================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =======================================
# TRAIN / VAL / TEST SPLIT
# =======================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# =======================================
# GRID SEARCH WITH TQDM
# =======================================
params = {
    "max_depth": [4, 6, 8, 10, 12, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10],
}

param_grid = list(ParameterGrid(params))
print(f"Grid Search: {len(param_grid)} combinations")

best_score = -np.inf
best_params = None

start = time.time()

for p in tqdm(param_grid, desc="Grid Search", unit="combo"):
    clf = DecisionTreeClassifier(**p)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    mean_score = f1_score(y_val, y_val_pred, average="macro")
    if mean_score > best_score:
        best_score = mean_score
        best_params = p

elapsed = time.time() - start
print(f"\nGrid Search done in {elapsed:.2f} seconds")
print(f"Best F1_macro: {best_score:.4f}")
print(f"Best params: {best_params}\n")

# =======================================
# TRAIN FINAL MODEL
# =======================================
model = DecisionTreeClassifier(**best_params)
model.fit(X_train, y_train)

# =======================================
# EVALUATION
# =======================================
y_pred = model.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== FEATURE IMPORTANCE ===")
for name, importance in zip(feature_cols, model.feature_importances_):
    print(f"{name:20s} : {importance:.4f}")

# =======================================
# SAVE MODEL
# =======================================
joblib.dump({
    "model": model,
    "label_encoder": label_encoder,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "best_params": best_params
}, MODEL_PATH)

print(f"\n[SAVED] Model saved to {MODEL_PATH}")

print("[INFO] Generating decision tree diagram...")
import export_decision_tree
export_decision_tree.main()
import export_reports
export_reports.main()
