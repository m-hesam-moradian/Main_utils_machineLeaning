# =========================================================
# SHAP Feature Importance → DataFrame + Clipboard
# =========================================================
# Requirements:
#   pip install pandas numpy shap scikit-learn matplotlib openpyxl
# =========================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# -------------------- 1. Load dataset --------------------
sheet_name = "DATA_Shuffled"
file_path = r"C:\Users\Sam\Desktop\BSE. No.14-Dataset.xlsx"

df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()

# Automatically detect target column (last column)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# -------------------- 2. Split & scale data --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- 3. Train model --------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# -------------------- 4. Compute SHAP values --------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# -------------------- 5. Summarize feature importance --------------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

# -------------------- 6. Plot SHAP summary --------------------
plt.title("SHAP Summary Plot (Mean Absolute Values)")
shap.summary_plot(shap_values, X_test, plot_type="bar")

# -------------------- 7. Copy results to clipboard --------------------
importance_df.to_clipboard(index=False)
print("✅ SHAP feature importance copied to clipboard!")
print(importance_df)
