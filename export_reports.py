import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ml_utils import MODEL_PATH, CSV_PATH, load_model, load_dataset, split_dataset, predict_with_model

OUT_CM = "static/uploads/confusion_matrix.png"
OUT_CR = "static/uploads/classification_report.png"

os.makedirs(os.path.dirname(OUT_CM), exist_ok=True)

def main():
    # load model
    model, label_encoder, scaler, feature_cols, _ = load_model(MODEL_PATH)

    # load dataset
    df, X, y_raw, y, used_cols, _ = load_dataset(CSV_PATH, feature_cols)

    # split dataset
    _, _, X_test, _, _, y_test = split_dataset(X, y)

    # predict
    y_pred = predict_with_model(model, scaler, X_test)
    class_names = label_encoder.classes_

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    # plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_CM, dpi=150)
    plt.close()
    print(f"[OK] Saved {OUT_CM}")

    # classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = (
        pd.DataFrame(report)
        .transpose()
        .loc[class_names.tolist() + ["macro avg", "weighted avg"], ["precision", "recall", "f1-score"]]
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df, annot=True, cmap="Greens", fmt=".2f")
    # plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig(OUT_CR, dpi=150)
    plt.close()
    print(f"[OK] Saved {OUT_CR}")

if __name__ == "__main__":
    main()
