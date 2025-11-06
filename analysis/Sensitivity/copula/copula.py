from couples_sensitivity_analysis import couples_sensitivity_analysis
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import QuantileRegressor

# Load the dataset
dt = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx", sheet_name="Data_after_KFold_QR"
)
target_column = "Remaining Useful Life "
X = dt.drop(target_column, axis=1)

y = dt[target_column]

features = X.columns

# Train the XGBoost model
model = QuantileRegressor(quantile=0.5, alpha=0.01, solver="highs")
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
copula.to_clipboard()


