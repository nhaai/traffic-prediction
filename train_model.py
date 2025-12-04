import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

CSV_PATH = "dataset_features.csv"
MODEL_PATH = "model.pkl"

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(CSV_PATH)

feature_cols = [
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "total_vehicles",
    "bbox_area_ratio",
    "mean_bbox_area",
    "max_bbox_area",
    "brightness",
    "sharpness",
    "edge_density",
]

X = df[feature_cols].values
y_raw = df["label"]

# =====================================================
# LABEL ENCODING
# =====================================================
le = LabelEncoder()
y = le.fit_transform(y_raw)

# =====================================================
# NORMALIZATION (Decision Tree does not require it,
# but helpful for consistency and future ML extensions)
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# TRAIN / VAL / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# =====================================================
# MODEL + GRID SEARCH
# =====================================================
params = {
    "max_depth": [4, 6, 8, 10, 12, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeClassifier(),
    params,
    scoring="f1_macro",
    cv=3,
    verbose=1
)

grid.fit(X_train, y_train)
model = grid.best_estimator_

print("\n=== BEST HYPERPARAMETERS ===")
print(grid.best_params_)

# =====================================================
# EVALUATE MODEL
# =====================================================
print("\n=== CLASSIFICATION REPORT ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
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
