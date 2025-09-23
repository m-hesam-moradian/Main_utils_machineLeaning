from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Step 1: Load data
sheet_name = "LabelEncoded"
data = pd.read_excel(
    r"D:\ML\Main_utils\Task\GLEMETA_MADDPG_Final_IoT_MEC_UAV_Dataset.xlsx",
    sheet_name=sheet_name,
)

X = data.drop("offload_ratio", axis=1)
y = data["offload_ratio"]

feature_names = X.columns.tolist()

# مدل
model = RandomForestClassifier()
model.fit(X, y)

# نمایش اهمیت فیچرها
importances = model.feature_importances_
for name, score in zip(feature_names, importances):
    print(f"Feature: {name}, Importance: {score:.4f}")
