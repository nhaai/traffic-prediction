import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import export_graphviz

MODEL_PATH = "model.pkl"
OUT_PDF = "static/decision_tree.pdf"
OUT_PNG = "static/decision_tree.png"
OUT_TXT = "static/decision_tree.txt"

# ============================================================
# LOAD MODEL
# ============================================================
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found: {MODEL_PATH}", file=sys.stderr)
    sys.exit(2)

data = joblib.load(MODEL_PATH)
model = data.get("model", None)
label_encoder = data.get("label_encoder", None)
feature_cols = data.get("feature_cols", None)

if model is None or label_encoder is None or feature_cols is None:
    print("[ERROR] model.pkl missing required keys ('model','label_encoder','feature_cols')", file=sys.stderr)
    sys.exit(2)

if not isinstance(model, DecisionTreeClassifier):
    raise TypeError("Loaded model is not a DecisionTreeClassifier.")

classes = list(label_encoder.classes_)
print("[INFO] Loaded model and metadata.")
print(f"[INFO] Tree nodes: {model.tree_.node_count}, max_depth: {model.tree_.max_depth}")

# ============================================================
# Export textual rules
# ============================================================
print(f"[INFO] Exporting text rules → {OUT_TXT}")

try:
    text_rules = export_text(model, feature_names=feature_cols)
except TypeError:
    text_rules = export_text(model, feature_names=feature_cols, max_depth=9999)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(text_rules)

print("[OK] Text rules saved.")

# ============================================================
# Try Graphviz first (best quality vector)
# ============================================================
use_graphviz = False
try:
    import graphviz

    use_graphviz = True
except Exception:
    use_graphviz = False

if use_graphviz:
    try:
        print("[INFO] Graphviz Python package found. Attempting graphviz export (vector).")
        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_cols,
            class_names=classes,
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph = graphviz.Source(dot_data)
        # Render PDF (vector) and PNG (raster) via graphviz (requires dot in PATH)
        graph.format = "pdf"
        graph.render(filename=OUT_PDF.replace(".pdf", ""), cleanup=True)
        print(f"[OK] Graphviz PDF written: {OUT_PDF}")

        graph.format = "png"
        # use a high resolution by letting graphviz decide; output filename without extension
        graph.render(filename=OUT_PNG.replace(".png", ""), cleanup=True)
        print(f"[OK] Graphviz PNG written: {OUT_PNG}")

        # Done
        sys.exit(0)

    except Exception as e:
        print("[WARN] Graphviz export failed or 'dot' not available. Falling back to matplotlib.", file=sys.stderr)
        print("[WARN]", str(e), file=sys.stderr)
        use_graphviz = False

# ============================================================
# Compute a sensible figsize proportional to tree size (node_count, depth)
# ============================================================
n_nodes = getattr(model.tree_, "node_count", None)
max_depth = getattr(model.tree_, "max_depth", None)

if n_nodes is None:
    n_nodes = 1
if max_depth is None or max_depth <= 0:
    max_depth = 1

# Heuristics to compute figure size:
fig_width = min(max(12.0, n_nodes / 4.0), 160.0)
fig_height = min(max(6.0, max_depth * 2.8), 120.0)

# Cap to avoid absurd sizes
fig_width = float(fig_width)
fig_height = float(fig_height)

print(f"[INFO] Matplotlib fallback: figsize = ({fig_width:.1f}, {fig_height:.1f}), dpi=300")

plt.figure(figsize=(fig_width, fig_height))
plt.rcParams["figure.facecolor"] = "white"

# ============================================================
# Plot full tree (no max_depth) to capture all nodes
# ============================================================
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

    # Save vector PDF
    plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    print(f"[OK] Matplotlib PDF saved: {OUT_PDF}")

    # Save raster PNG high-res (increase dpi for better readability)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[OK] Matplotlib PNG saved: {OUT_PNG}")
except Exception as e:
    print("[ERROR] Failed to render tree with matplotlib:", file=sys.stderr)
    print(e, file=sys.stderr)
finally:
    plt.close()

print("[DONE] All exports complete.")
