import pandas as pd
import joblib

csv = pd.read_csv("dataset_features.csv")
print(csv["label"].value_counts())
print(csv.groupby("cam_id")["label"].value_counts())
print("\n")
model = joblib.load("model.pkl")
print(model.keys())
print(model["best_params"])
print(model["feature_cols"])
print(model["model"])
