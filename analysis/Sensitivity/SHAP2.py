# =========================================================
# SHAP Sensitivity Analysis → Table + Dependence Plots (Saved as Images)
# Modified for XGBoost (XGBRegressor)
# =========================================================
# Requirements:
#   pip install pandas numpy shap scikit-learn matplotlib openpyxl xgboost
# =========================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # <-- Changed to XGBoost
import os

# -------------------- 1. Load dataset --------------------

sheet_name = "Data_after_KFold_HR"
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Automatically detect target column (last column)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# -------------------- 2. Split data --------------------
# Note: StandardScaler was removed because XGBoost does not need scaled data.
# This makes your SHAP plots much easier to read (they will use original units).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- 3. Train model --------------------
# Changed to XGBoost Regressor
model = XGBRegressor(    n_estimators=100,       # keep moderate
    max_depth=3,      )
    
model.fit(X_train, y_train)

# -------------------- 4. Compute SHAP values --------------------
# Changed to TreeExplainer (optimized for XGBoost)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# -------------------- 5. Build Sensitivity Table (for Clipboard) --------------------
print("Calculating Sensitivity Metrics...")

metrics_list = []

for i, feature_name in enumerate(X.columns):
    # Sensitivity: Mean Absolute SHAP
    mean_abs_shap = np.abs(shap_values[:, i]).mean()
    
    # Dependence Ranges: Min/Max SHAP
    max_shap = np.max(shap_values[:, i])
    min_shap = np.min(shap_values[:, i])
    impact_range = max_shap - min_shap
    
    # Dependence Trend: Correlation (Feature Value vs SHAP Value)
    feature_vals = X_test[feature_name].values
    correlation = np.corrcoef(feature_vals, shap_values[:, i])[0, 1]

    metrics_list.append({
        "Feature": feature_name,
        "Mean_Abs_SHAP": mean_abs_shap,
        "Max_SHAP": max_shap,
        "Min_SHAP": min_shap,
        "Impact_Range": impact_range,
        "Feature_Correlation": correlation
    })

importance_df = pd.DataFrame(metrics_list)
importance_df = importance_df.sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

# Format Table
importance_df["Mean_Abs_SHAP"] = importance_df["Mean_Abs_SHAP"].map('{:.4f}'.format)
importance_df["Max_SHAP"] = importance_df["Max_SHAP"].map('{:.4f}'.format)
importance_df["Min_SHAP"] = importance_df["Min_SHAP"].map('{:.4f}'.format)
importance_df["Feature_Correlation"] = importance_df["Feature_Correlation"].map('{:.4f}'.format)

# -------------------- 6. Generate and Save Plots --------------------
# Create a folder to save plots
output_dir = "SHAP_Plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\nSaving plots to folder: {output_dir}")

# Plot 1: SHAP Summary Plot (Beeswarm)
# This shows the overall distribution of feature impacts
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.savefig(os.path.join(output_dir, "01_Summary_Beeswarm.png"), bbox_inches='tight', dpi=300)
plt.close() # Close figure to free memory
print("Saved: 01_Summary_Beeswarm.png")

# Plot 2: SHAP Dependence Plots
# These show how specific features impact the model output
top_n_features = 3  # Generate dependence plots for top 3 features
for i, feature in enumerate(importance_df["Feature"].head(top_n_features)):
    plt.figure()
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    plt.savefig(os.path.join(output_dir, f"02_Dependence_{feature}.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: 02_Dependence_{feature}.png")

# -------------------- 7. Report Table --------------------
print("\n" + "="*80)
print(" SHAP SENSITIVITY TABLE (Copied to Clipboard) ".center(80, "="))
print("="*80)
print(importance_df.to_string())
print("="*80)

# -------------------- 8. Copy to clipboard --------------------
importance_df.to_clipboard(index=False)
print("\n✅ Table copied to clipboard! Check 'SHAP_Plots' folder for images.")