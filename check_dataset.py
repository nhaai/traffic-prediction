import pandas as pd
df = pd.read_csv("dataset_features.csv")
print(df["label"].value_counts())
print(df.groupby("cam_id")["label"].value_counts())
