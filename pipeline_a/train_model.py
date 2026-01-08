import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from ml_utils import load_dataset, fit_scaler, save_model

MODEL_PATH = "models/hc.pkl"
CSV_PATH = "pipeline_a/dataset_features.csv"
CSV_TRAIN_PATH = "pipeline_a/train.csv"
CSV_VAL_PATH = "pipeline_a/val.csv"
CSV_TEST_PATH = "pipeline_a/test.csv"

# =======================================
# LOAD DATASET
# =======================================
df, X, y_raw, y, used_cols, label_encoder = load_dataset(CSV_PATH)
print("Loaded CSV rows:", len(df))
print(df.head())
print("\nUsing feature columns:", used_cols, "\n")

jam_label = label_encoder.transform(["congested"])[0]

# =======================================
# SPLIT DATA
# =======================================
_, X_train, _, y_train, _, _, = load_dataset(CSV_TRAIN_PATH, feature_cols=used_cols, label_encoder=label_encoder)
_, X_val, _, y_val, _, _, = load_dataset(CSV_VAL_PATH, feature_cols=used_cols, label_encoder=label_encoder)
_, X_test, _, y_test, _, _, = load_dataset(CSV_TEST_PATH, feature_cols=used_cols, label_encoder=label_encoder)

# =======================================
# ADABOOST FOR FEATURE IMPORTANCE
# =======================================
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
y_train_jam = np.asarray(y_train == jam_label, dtype=int)
ada.fit(X_train, y_train_jam)
ada_importance = ada.feature_importances_

if np.all(ada_importance == 0):
    print("[WARN] AdaBoost importance all zero, fallback to all features")
    selected_features = used_cols
else:
    importance_df = pd.DataFrame({
        "feature": used_cols,
        "importance": ada_importance
    }).sort_values("importance", ascending=False)
    TOP_K = min(12, len(used_cols))
    selected_features = importance_df.head(TOP_K)["feature"].tolist()

selected_idx = [used_cols.index(f) for f in selected_features]

X_train = X_train[:, selected_idx]
X_val   = X_val[:, selected_idx]
X_test  = X_test[:, selected_idx]

used_cols = selected_features
print("Selected features:", used_cols)

# =======================================
# SCALER
# =======================================
scaler, X_train = fit_scaler(X_train)
X_val  = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =======================================
# GRID SEARCH
# =======================================
params = {
    "max_depth": [4, 6, 8, 10, 12, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"],
}

param_grid = [
    dict(max_depth=d, criterion=c, min_samples_split=s, class_weight=w)
    for d in params["max_depth"]
    for c in params["criterion"]
    for s in params["min_samples_split"]
    for w in params["class_weight"]
]

best_score = -np.inf
best_params = None

print(f"Grid Search: {len(param_grid)} combinations")
start = time.time()

for p in param_grid:
    clf = DecisionTreeClassifier(random_state=42, **p)
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    y_val_jam = np.asarray(y_val == jam_label, dtype=int)
    y_pred_jam = np.asarray(y_val_pred == jam_label, dtype=int)
    f1 = f1_score(y_val_jam, y_pred_jam, pos_label=1)

    if f1 > best_score:
        best_score = f1
        best_params = p

elapsed = time.time() - start
print(f"Completed in {elapsed:.2f}s")
print("Best params:", best_params)
print("Best score:", best_score)

# =======================================
# TRAIN MODEL
# =======================================
model = DecisionTreeClassifier(random_state=42, **best_params)
model.fit(X_train, y_train)

# =======================================
# EVALUATION
# =======================================
y_pred = model.predict(X_test)

print("\n=== CLASSIFICATION REPORT (3-CLASS) ===")
print(classification_report(y_test, y_pred))

print("=== CONFUSION MATRIX (3-CLASS) ===")
print(confusion_matrix(y_test, y_pred))

y_test_jam = np.asarray(y_test == jam_label, dtype=int)
y_pred_jam = np.asarray(y_pred == jam_label, dtype=int)
print("\nF1_jam:", f1_score(y_test_jam, y_pred_jam, pos_label=1))
print("=== CONFUSION MATRIX (JAM) ===")
print(confusion_matrix(y_test_jam, y_pred_jam, labels=[0, 1]))

print("\n=== FEATURE IMPORTANCE ===")
for name, importance in zip(used_cols, model.feature_importances_):
    print(f"{name:20s} : {importance:.4f}")

# =======================================
# SAVE MODEL
# =======================================
save_model(MODEL_PATH, model, label_encoder, scaler, used_cols, best_params)

# =======================================
# GENERATE TREE + REPORTS
# =======================================
import export_decision_tree
import export_reports

export_decision_tree.main()
export_reports.main()
