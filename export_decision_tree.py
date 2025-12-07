import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

EXPORT_TXT = False
EXPORT_PDF = False
EXPORT_PNG = True

MODEL_PATH = "model.pkl"
OUT_PDF = "static/decision_tree.pdf"
OUT_PNG = "static/decision_tree.png"
OUT_TXT = "static/decision_tree.txt"

os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}", file=sys.stderr)
        return

    data = joblib.load(MODEL_PATH)
    model = data.get("model", None)
    label_encoder = data.get("label_encoder", None)
    feature_cols = data.get("feature_cols", None)

    if model is None or label_encoder is None or feature_cols is None:
        print("[ERROR] model.pkl missing required keys ('model','label_encoder','feature_cols')", file=sys.stderr)
        return

    if not isinstance(model, DecisionTreeClassifier):
        raise TypeError("Loaded model is not a DecisionTreeClassifier.")

    classes = list(label_encoder.classes_)
    print(f"[INFO] Tree nodes: {model.tree_.node_count}, max_depth: {model.tree_.max_depth}")

    # export text rules
    if EXPORT_TXT:
        print(f"[INFO] Exporting text rules → {OUT_TXT}")
        try:
            text_rules = export_text(model, feature_names=feature_cols, max_depth=9999)
        except TypeError:
            text_rules = export_text(model, feature_names=feature_cols)

        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write(text_rules)

        print("[OK] Text rules saved.")

    # try Graphviz first
    try:
        import graphviz
        use_graphviz = True
    except Exception:
        use_graphviz = False

    if use_graphviz:
        from shutil import which
        if which("dot") is None:
            print("[WARN] 'graphviz' package found but system 'dot' missing. Falling back to Matplotlib.")
            use_graphviz = False

    if use_graphviz and (EXPORT_PDF or EXPORT_PNG):
        try:
            dot_data = export_graphviz(
                model,
                out_file=None,
                feature_names=feature_cols,
                class_names=classes,
                filled=True,
                rounded=True,
                special_characters=False
            )
            graph = graphviz.Source(dot_data)

            if EXPORT_PDF:
                graph.render(OUT_PDF, cleanup=True, format="pdf")
                print(f"[OK] Saved: {OUT_PDF}")
            if EXPORT_PNG:
                graph.render(OUT_PNG, cleanup=True, format="png")
                print(f"[OK] Saved: {OUT_PNG}")

            return
        except Exception as e:
            print("[WARN] Graphviz export failed, using Matplotlib fallback.")
            print("[WARN]", str(e))

    # compute a sensible figsize proportional to tree size (node_count, depth)
    if not (EXPORT_PDF or EXPORT_PNG):
        print("[INFO] PDF/PNG export disabled.")
        return

    n_nodes = getattr(model.tree_, "node_count", 1)
    max_depth = getattr(model.tree_, "max_depth", 1)

    fig_width = min(max(12.0, n_nodes / 4.0), 160.0)
    fig_height = min(max(6.0, max_depth * 2.0), 120.0)

    # print(f"[INFO] Matplotlib figsize = ({fig_width:.1f}, {fig_height:.1f}), dpi=150")

    plt.figure(figsize=(fig_width, fig_height), dpi=150)
    plt.rcParams["figure.facecolor"] = "white"

    # plot full tree (no max_depth) to capture all nodes
    try:
        tree.plot_tree(
            model,
            feature_names=feature_cols,
            class_names=classes,
            filled=True,
            rounded=True,
            fontsize=8,
            proportion=False
        )
        plt.tight_layout()

        if EXPORT_PDF:
            plt.savefig(OUT_PDF, dpi=150, facecolor="white", bbox_inches="tight")
            print(f"[OK] Saved: {OUT_PDF}")
        if EXPORT_PNG:
            plt.savefig(OUT_PNG, dpi=150, facecolor="white", bbox_inches="tight")
            print(f"[OK] Saved: {OUT_PNG}")
    except Exception as e:
        print("[ERROR] Matplotlib rendering failed:", file=sys.stderr)
        print(e, file=sys.stderr)
    finally:
        plt.close()

if __name__ == "__main__":
    main()
