# =========================================================
# SHAP Sensitivity Analysis -> Table + Dependence Plots (Saved as Images)
# Modified for XGBoost (XGBClassifier) - Multiclass Fixed
# =========================================================
# Requirements:
#   pip install pandas numpy shap scikit-learn matplotlib openpyxl xgboost
# =========================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os
import warnings

# Ignore minor warnings for cleaner console output
warnings.filterwarnings('ignore')

# -------------------- 1. Load dataset --------------------

sheet_name = "Data_after_KFold_MLR(SMOTE-ENC)"
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Automatically detect target column (last column)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]
print(f"Target variable distribution:\n{y.value_counts()}\n")

# -------------------- 2. Split data --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- 3. Train model --------------------
# Note: max_depth=1000 is very large for XGBoost (default is 6), 
# but kept as-is to match your original script.
# model = XGBClassifier(
#     n_estimators=1000,       
#     max_depth=1000,      
#     random_state=42
# )
model = XGBClassifier(
    n_estimators=1000,       
    max_depth=1000,      
    random_state=42
)
    
model.fit(X_train, y_train)

# -------------------- 4. Compute SHAP values --------------------
print("Computing SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(model)

# FIX 1: check_additivity=False stops the rounding error crash
shap_values_raw = explainer.shap_values(X_test, check_additivity=False)

# FIX 2: Handle Multiclass SHAP output (3D array or List)
# Multiclass models give SHAP values for EACH class. 
# We specify which class we want to generate the table/plots for below:
TARGET_CLASS = 0  # <--- Change to 1 or 2 to analyze the other failure modes

if isinstance(shap_values_raw, list):
    shap_values = shap_values_raw[TARGET_CLASS]
elif len(shap_values_raw.shape) == 3:
    shap_values = shap_values_raw[:, :, TARGET_CLASS]
else:
    shap_values = shap_values_raw # Fallback if it's binary

print(f"Extracted SHAP values for Class {TARGET_CLASS}.")

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
    
    # Handle edge cases where standard deviation is 0 to avoid NaNs
    if np.std(feature_vals) == 0 or np.std(shap_values[:, i]) == 0:
        correlation = 0.0
    else:
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
# Create a folder to save plots (FIX 3: Added 'r' for raw string)
output_dir = r"C:\Users\Sam\Desktop\ML\task\SHAP_Plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\nSaving plots to folder: {output_dir}")

# Plot 1: SHAP Summary Plot (Beeswarm)
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title(f"SHAP Summary (Class {TARGET_CLASS})", pad=20)
plt.savefig(os.path.join(output_dir, f"01_Summary_Beeswarm_Class{TARGET_CLASS}.png"), bbox_inches='tight', dpi=300)
plt.close() # Close figure to free memory
print(f"Saved: 01_Summary_Beeswarm_Class{TARGET_CLASS}.png")

# Plot 2: SHAP Dependence Plots
top_n_features = 3  # Generate dependence plots for top 3 features
for i, feature in enumerate(importance_df["Feature"].head(top_n_features)):
    plt.figure()
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    plt.title(f"SHAP Dependence: {feature} (Class {TARGET_CLASS})")
    plt.savefig(os.path.join(output_dir, f"02_Dependence_{feature}_Class{TARGET_CLASS}.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: 02_Dependence_{feature}_Class{TARGET_CLASS}.png")

# -------------------- 7. Report Table --------------------
print("\n" + "="*80)
print(f" SHAP SENSITIVITY TABLE - CLASS {TARGET_CLASS} (Copied to Clipboard) ".center(80, "="))
print("="*80)
print(importance_df.to_string())
print("="*80)

# -------------------- 8. Copy to clipboard --------------------
importance_df.to_clipboard(index=False)
print(f"\n✅ Table for Class {TARGET_CLASS} copied to clipboard! Check 'SHAP_Plots' folder for images.")