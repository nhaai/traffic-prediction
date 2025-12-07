import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "dataset_features.csv"
MODEL_PATH = "model.pkl"
OUT_PNG_1 = "static/confusion_matrix.png"
OUT_PNG_2 = "static/classification_report.png"

os.makedirs(os.path.dirname(OUT_PNG_1), exist_ok=True)

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found: {CSV_PATH}", file=sys.stderr)
        return
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}", file=sys.stderr)
        return

    df = pd.read_csv(CSV_PATH)
    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    label_encoder = saved["label_encoder"]
    scaler = saved["scaler"]
    feature_cols = saved["feature_cols"]

    X = df[feature_cols].values
    y_raw = df["label"].values
    y = label_encoder.transform(y_raw)

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # validation split is irrelevant here because we only evaluate test set
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    class_names = label_encoder.classes_

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT_PNG_1, dpi=300)
    plt.close()

    print(f"[OK] Saved: {OUT_PNG_1}")

    # classification report heatmap
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )

    df_report = pd.DataFrame(report).transpose()

    rows = class_names.tolist() + ["macro avg", "weighted avg"]
    df_plot = df_report.loc[rows, ["precision", "recall", "f1-score"]]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df_plot,
        annot=True,
        cmap="Greens",
        fmt=".2f",
        linewidths=.5
    )
    plt.title("Classification Report Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_PNG_2, dpi=300)
    plt.close()

    print(f"[OK] Saved: {OUT_PNG_2}")

if __name__ == "__main__":
    main()
