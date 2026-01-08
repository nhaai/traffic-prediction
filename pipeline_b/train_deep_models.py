import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier

# =======================================
# LOAD DATASET
# =======================================
df = pd.read_csv("pipeline_b/dataset_deep_features.csv")

feature_cols = [c for c in df.columns if c.startswith("f") and c not in ["label", "cam_id", "filepath"]]
X = df[feature_cols].values
y_raw = df["label"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================================
# MODELS
# =======================================
models = {
    "rf": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ),
    "gb": GradientBoostingClassifier(
        n_estimators=200,
        random_state=42
    ),
    "ada": AdaBoostClassifier(
        n_estimators=200,
        random_state=42
    ),
    "xgb": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    ),
    "svm": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    ),
}

# =======================================
# TRAIN + EVAL
# =======================================
for name, model in models.items():
    print(f"\n=== Training {name.upper()} ===")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols
    }, f"models/{name}.pkl")
