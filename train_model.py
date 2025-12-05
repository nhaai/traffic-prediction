import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.metrics import f1_score
import joblib
from tqdm import tqdm
import time

CSV_PATH = "dataset_features.csv"
MODEL_PATH = "model.pkl"

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(CSV_PATH)
print("Loaded CSV rows:", len(df))
print(df.head())

feature_cols = [
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    # "total_vehicles",
    "bbox_area_ratio",
    "mean_bbox_area",
    "max_bbox_area",
    "brightness",
    "sharpness",
    "edge_density",
    "is_night",
]

X = df[feature_cols].values
y_raw = df["label"]

# =====================================================
# LABEL ENCODING
# =====================================================
le = LabelEncoder()
y = le.fit_transform(y_raw)

# =====================================================
# NORMALIZATION
# (Decision Tree does not require it, but OK for consistency)
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# =====================================================
# MANUAL GRID SEARCH WITH TQDM
# =====================================================
params = {
    "max_depth": [4, 6, 8, 10, 12, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10],
}

param_grid = list(ParameterGrid(params))
total_combos = len(param_grid)  # 6 * 2 * 3 = 36

print(f"Starting Grid Search: {total_combos} combinations × 3-fold CV")

best_score = -np.inf
best_params = None

start_time = time.time()

for p in tqdm(param_grid, desc="Grid Search", unit="combo"):
    clf = DecisionTreeClassifier(**p)

    # Use cross_val_score for proper 3-fold CV
    scores = cross_val_score(
        clf, X_train, y_train,
        cv=3,
        scoring="f1_macro",
        n_jobs=1  # keep predictable timing for tqdm
    )

    mean_score = scores.mean()

    if mean_score > best_score:
        best_score = mean_score
        best_params = p

elapsed = time.time() - start_time
print(f"\nGrid Search done in {elapsed:.2f} seconds")
print(f"Best F1_macro: {best_score:.4f}")
print(f"Best params: {best_params}\n")

# =====================================================
# TRAIN FINAL MODEL
# =====================================================
model = DecisionTreeClassifier(**best_params)
model.fit(X_train, y_train)

# =====================================================
# EVALUATE
# =====================================================
print("\n=== CLASSIFICATION REPORT ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== FEATURE IMPORTANCE ===")
for name, importance in zip(feature_cols, model.feature_importances_):
    print(f"{name:20s} : {importance:.4f}")

# =====================================================
# SAVE MODEL
# =====================================================
joblib.dump({
    "model": model,
    "label_encoder": le,
    "scaler": scaler,
    "feature_cols": feature_cols
}, MODEL_PATH)

print(f"\n[SAVED] Model saved to {MODEL_PATH}\n")
