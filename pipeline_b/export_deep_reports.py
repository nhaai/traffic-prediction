import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

MODELS = ["rf", "gb", "ada"]

df = pd.read_csv("dataset_deep_features.csv")
feature_cols = [c for c in df.columns if c.startswith("f")]

X = df[feature_cols].values
y_raw = df["label"].values

results = []

for name in MODELS:
    data = joblib.load(f"pipeline_b/deep_models/{name}.pkl")
    model = data["model"]
    scaler = data["scaler"]
    le = data["label_encoder"]

    y = le.transform(y_raw)
    Xs = scaler.transform(X)

    y_pred = model.predict(Xs)
    acc = accuracy_score(y, y_pred)

    results.append({
        "model": name,
        "accuracy": acc
    })

print("\n=== PIPELINE B RESULTS ===")
print(pd.DataFrame(results))
