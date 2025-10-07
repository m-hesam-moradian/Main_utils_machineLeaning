import sys

sys.path.append("/Users/pouya/Desktop/machin_learning_work")
from couples_sensitivity_analysis import couples_sensitivity_analysis
import pandas as pd
from xgboost import XGBRegressor

# Load the dataset
dt = pd.read_excel("./5G6G_Optimization_Dataset.xlsx", sheet_name="LGBR_DATA")

X = dt.drop("Throughput_Mbps", axis=1)

y = dt["Throughput_Mbps"]


features = X.columns

# Train the XGBoost model
model = XGBRegressor()
model.fit(X, y)

# Define pairs of features for sensitivity analysis
feature_pairs = [
    (features[i], features[j])
    for i in range(len(features))
    for j in range(len(features))
]

# Perform the couples sensitivity analysis
c = couples_sensitivity_analysis(model, X, y, feature_pairs, "mse", 40)

# Display the sensitivity report
print(c)
