from couples_sensitivity_analysis import couples_sensitivity_analysis
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
dt = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    sheet_name="Data_after_KFold_ETR"  # Adjust sheet name as needed,
)
 
target_column = dt.columns[-1]
X = dt.drop(target_column, axis=1)
y = dt[target_column]

features = X.columns
from xgboost import XGBRegressor

# 🌲 Random Forest Classifier (anti-overfitting setup)
model = XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
)

model.fit(X, y)

# Define pairs of features for sensitivity analysis
feature_pairs = [
    (features[i], features[j])
    for i in range(len(features))
    for j in range(len(features))
]

# Perform the couples sensitivity analysis
copula = couples_sensitivity_analysis(
    model,
    X,
    y,
    feature_pairs,
    "mse",  # Mean Squared Error for regression
    40
)

# Display the sensitivity report
print(copula)
copula.to_clipboard()