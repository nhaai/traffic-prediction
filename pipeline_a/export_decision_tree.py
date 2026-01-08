import os
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz
from ml_utils import load_model

MODEL_PATH = "models/hc.pkl"
EXPORT_PDF = False
EXPORT_PNG = True
OUT_PDF = "static/uploads/00_decision_tree.pdf"
OUT_PNG = "static/uploads/00_decision_tree.png"

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# =======================================
# MAIN
# =======================================
def main():
    model, label_encoder, scaler, feature_cols, _ = load_model(MODEL_PATH)
    print(f"[INFO] Tree nodes: {model.tree_.node_count}, depth: {model.tree_.max_depth}")

    # try graphviz first
    try:
        import graphviz
        from shutil import which

        if which("dot") is None:
            raise RuntimeError("dot not found")

        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_cols,
            class_names=[str(c) for c in label_encoder.classes_],
            filled=True,
            rounded=True,
            special_characters=False
        )

        graph = graphviz.Source(dot_data)

        if EXPORT_PDF:
            graph.render(OUT_PDF, cleanup=True, format="pdf")
            print("[OK] Saved PDF")

        if EXPORT_PNG:
            graph.render(OUT_PNG, cleanup=True, format="png")
            print("[OK] Saved PNG")

        return
    except (ModuleNotFoundError, OSError, RuntimeError):
        pass

    # fallback: matplotlib
    plt.figure(figsize=(20, 12), dpi=200)
    tree.plot_tree(
        model,
        feature_names=feature_cols,
        class_names=[str(c) for c in label_encoder.classes_],
        filled=True,
        fontsize=8
    )
    plt.tight_layout()

    if EXPORT_PNG:
        plt.savefig(OUT_PNG, dpi=200)
        print("[OK] Saved decision tree as png (matplotlib)")

    if EXPORT_PDF:
        plt.savefig(OUT_PDF, dpi=200)
        print("[OK] Saved decision tree as pdf (matplotlib)")

    plt.close()

if __name__ == "__main__":
    main()
