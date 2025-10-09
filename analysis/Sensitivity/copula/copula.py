from couples_sensitivity_analysis import couples_sensitivity_analysis
import pandas as pd
from xgboost import XGBRegressor

# Load the dataset
dt = pd.read_excel(
    r"D:\ML\Main_utils\task\Resource_utilization.xlsx", sheet_name="Data_after_KFold"
)
target_column = "cpu_utilization"
X = dt.drop(target_column, axis=1)

y = dt[target_column]


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
copula = couples_sensitivity_analysis(model, X, y, feature_pairs, "mse", 40)

# Display the sensitivity report
print(copula)
