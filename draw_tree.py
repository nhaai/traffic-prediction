import joblib
import matplotlib.pyplot as plt
from sklearn import tree

MODEL_PATH = "model.pkl"
OUTPUT_FILE = "decision_tree.png"

# ============================================================
# LOAD MODEL & METADATA
# ============================================================
data = joblib.load(MODEL_PATH)
model = data["model"]
label_encoder = data["label_encoder"]
feature_cols = data["feature_cols"]

classes = list(label_encoder.classes_)  # ["congested", "free_flow", "moderate"]

print("[INFO] Loaded model and metadata.")

# ============================================================
# PLOT DECISION TREE
# ============================================================
plt.figure(figsize=(28, 14))

tree.plot_tree(
    model,
    feature_names=feature_cols,
    class_names=classes,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()

print(f"[DONE] Decision Tree saved to {OUTPUT_FILE}")
