import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from ml_utils import MODEL_PATH, CSV_PATH, load_model, load_dataset, load_split_dataset, predict_with_model

# =======================================
# LABEL DISTRIBUTION
# =======================================
def export_label_distribution(df, save_path):
    labels = df["label"].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(labels.values, labels=labels.index, autopct="%1.1f%%")
    plt.title("Label Distribution")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

# =======================================
# HISTOGRAM OF KEY FEATURES
# =======================================
def export_feature_histogram(df, feature, save_path):
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x=feature, hue="label", kde=False, alpha=0.6)
    plt.title(f"Histogram of {feature}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

# =======================================
# BOXPLOT OF FEATURE
# =======================================
def export_feature_boxplot(df, feature, save_path):
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="label", y=feature)
    plt.title(f"Boxplot of {feature} by label")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

# =======================================
# SCATTER PLOT OF TWO FEATURES
# =======================================
def export_scatter_2d(df, feature_x, feature_y, save_path):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df, x=feature_x, y=feature_y, hue="label", alpha=0.7
    )
    plt.title(f"{feature_x} vs {feature_y}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

# =======================================
# FEATURE IMPORTANCE BAR CHART
# =======================================
def export_feature_importance(model, feature_cols, save_path):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_cols)[idx], importances[idx])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (Decision Tree)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

# =======================================
# CONFUSION MATRIX (3-CLASS)
# =======================================
def export_confusion_matrix_plot(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)

    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (3-Class)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

# =======================================
# CLASSIFICATION REPORT (3-CLASS)
# =======================================
def export_classification_report_plot(y_test, y_pred, save_path):
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(8, 5))
    sns.heatmap(df.iloc[:-1, :], annot=True, cmap="Greens", fmt=".2f")
    plt.title("Classification Report (3-Class)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


# ======================================================
# CONFUSION MATRIX (JAM / NOT JAM)
# ======================================================
def export_confusion_matrix_jam(y_test, y_pred, jam_label, save_path):
    y_test_jam = np.asarray(y_test == jam_label, dtype=int)
    y_pred_jam = np.asarray(y_pred == jam_label, dtype=int)

    cm = confusion_matrix(y_test_jam, y_pred_jam, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=["Not Jam", "Jam"]
    )

    plt.figure(figsize=(5, 4))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Jam Detection)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

# ======================================================
# CLASSIFICATION REPORT (JAM / NOT JAM)
# ======================================================
def export_classification_report_jam(y_test, y_pred, jam_label, save_path):
    y_test_jam = np.asarray(y_test == jam_label, dtype=int)
    y_pred_jam = np.asarray(y_pred == jam_label, dtype=int)

    report = classification_report(
        y_test_jam,
        y_pred_jam,
        target_names=["Not Jam", "Jam"],
        output_dict=True
    )

    df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(6, 4))
    sns.heatmap(df.iloc[:-1, :], annot=True, cmap="Oranges", fmt=".2f")
    plt.title("Classification Report (Jam Detection)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

# =======================================
# MAIN
# =======================================
def main():
    # load model
    model, label_encoder, scaler, feature_cols, _ = load_model(MODEL_PATH)

    # load dataset
    df, X, y_raw, y, used_cols, _ = load_dataset(CSV_PATH, feature_cols)
    _, X_test, _, y_test, _, _ = load_split_dataset("test", feature_cols)

    # predict
    y_pred = predict_with_model(model, scaler, X_test)

    # export reports
    export_label_distribution(df, "static/uploads/01_label_distribution.png")

    key_features = ["edge_density", "brightness", "sharpness", "bbox_area_ratio"]
    for f in key_features:
        if f in df.columns:
            export_feature_histogram(df, f, f"static/uploads/02_histogram_{f}.png")

    for f in key_features:
        if f in df.columns:
            export_feature_boxplot(df, f, f"static/uploads/03_boxplot_{f}.png")

    f1, f2 = key_features[:2]
    export_scatter_2d(df, f1, f2, "static/uploads/04_scatter_features.png")
    export_feature_importance(model, feature_cols, "static/uploads/05_feature_importance.png")

    export_confusion_matrix_plot(y_test, y_pred, "static/uploads/06_confusion_matrix_3class.png")
    export_classification_report_plot(y_test, y_pred, "static/uploads/07_classification_report_3class.png")

    jam_label = label_encoder.transform(["congested"])[0]
    export_confusion_matrix_jam(y_test, y_pred, jam_label, "static/uploads/06_confusion_matrix.png")
    export_classification_report_jam(y_test, y_pred, jam_label, "static/uploads/07_classification_report.png")

    print(f"[OK] Saved all reports")

if __name__ == "__main__":
    main()
