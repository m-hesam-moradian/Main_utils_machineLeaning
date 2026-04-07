import pandas as pd
import numpy as np
from xgboost import XGBRegressor  # Switched to XGBoost
from sklearn.model_selection import train_test_split
from SALib.sample import saltelli
from SALib.analyze import sobol

# --- Load dataset ---
sheet_name = "Data_after_KFold_RFC"
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"


df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()
target_column = df.columns[-1]

# --- Feature Selection ---
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Train XGBoost model (The "Best Model") ---
model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# --- Define problem space for SALib ---
feature_names = X.columns.tolist()
bounds = [[float(X[col].min()), float(X[col].max())] for col in feature_names]

problem = {
    "num_vars": len(feature_names),
    "names": feature_names,
    "bounds": bounds
}

# --- Generate samples ---
# 1024 is the base sample; total samples will be N * (D + 2)
param_values = saltelli.sample(problem, 1024, calc_second_order=False)

# --- Predict using XGBoost ---
# Convert samples to DataFrame to avoid feature name warnings in XGBoost
param_df = pd.DataFrame(param_values, columns=feature_names)
predictions = model.predict(param_df)

# --- Run SOBOL analysis ---
sobol_results = sobol.analyze(problem, predictions, calc_second_order=False, print_to_console=False)

# --- Format results ---
# S1: First-order sensitivity (contribution of the feature alone)
# ST: Total-order sensitivity (contribution including interactions with other features)
sensitivity_df = pd.DataFrame({
    "Feature": feature_names,
    "S1": sobol_results["S1"],
    "ST": sobol_results["ST"],  # Added Total Index as it's useful for non-linear models
    "S1_conf": sobol_results["S1_conf"]
}).sort_values(by="ST", ascending=False).reset_index(drop=True)

# --- Export to clipboard ---
sensitivity_df.to_clipboard(index=False)
print("Sobol Sensitivity Results (Sorted by Total Index ST):")
print(sensitivity_df)