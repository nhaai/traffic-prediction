import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from ml_utils import CSV_PATH, MODEL_PATH, load_dataset, fit_scaler, split_dataset, save_model

# =======================================
# LOAD DATASET
# =======================================
df, X, y_raw, y, used_cols, label_encoder = load_dataset(CSV_PATH)
print("Loaded CSV rows:", len(df))
print(df.head())
print("\nUsing feature columns:", used_cols, "\n")

# =======================================
# SCALER
# =======================================
scaler, X_scaled = fit_scaler(X)

# =======================================
# SPLIT DATA
# =======================================
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_scaled, y)

# =======================================
# GRID SEARCH
# =======================================
params = {
    "max_depth": [4, 6, 8, 10, 12, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10],
}

param_grid = [
    dict(max_depth=d, criterion=c, min_samples_split=s)
    for d in params["max_depth"]
    for c in params["criterion"]
    for s in params["min_samples_split"]
]

best_score = -np.inf
best_params = None

print(f"Grid Search: {len(param_grid)} combinations")
start = time.time()

for p in param_grid:
    clf = DecisionTreeClassifier(**p)
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_val_pred, average="macro")

    if f1 > best_score:
        best_score = f1
        best_params = p

elapsed = time.time() - start
print(f"Completed in {elapsed:.2f}s")
print("Best params:", best_params)
print("Best F1_macro:", best_score)

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
print(classification_report(y_test, y_pred))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

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
