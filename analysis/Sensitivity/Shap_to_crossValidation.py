import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor  # <-- Changed to Regressor!
from sklearn.model_selection import KFold
import warnings

# Suppress warnings for a clean console
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
SHEET_NAME = "KfoldXGBR" # <-- Updated to your current sheet

print(f"Loading data from sheet: '{SHEET_NAME}'...")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]
features = X.columns.tolist()

# ---------------------------------------------------------
# 2. Setup 5-Fold Cross Validation (UNSHUFFLED)
# ---------------------------------------------------------
# n_splits=5 and shuffle=False means it cuts the data into 5 exact chunks 
kf = KFold(n_splits=5, shuffle=False)

# Changed to XGBoost Regressor for continuous target data
model = XGBRegressor(n_estimators=100, random_state=42)

# Dictionary to store the Mean Absolute SHAP values for each feature
shap_results = {feature: [] for feature in features}

# ---------------------------------------------------------
# 3. Train and Calculate SHAP per Fold
# ---------------------------------------------------------
print("Running 5-Fold Cross Validation (Regression) and calculating SHAP values...")

for fold_idx, (train_index, test_index) in enumerate(kf.split(X), 1):
    print(f"  -> Processing Test Phase (Fold) {fold_idx}...")
    
    # Split the data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the Regression model
    model.fit(X_train, y_train)
    
    # Calculate SHAP values specifically for this test phase
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test, check_additivity=False)
    
    # For Regression, SHAP values are a clean 2D array, so we just take the mean across the samples (axis=0)
    fold_shap_mean = np.abs(shap_values_raw).mean(axis=0)
        
    # Store the SHAP values for this fold
    for i, feature in enumerate(features):
        shap_results[feature].append(fold_shap_mean[i])

# ---------------------------------------------------------
# 4. Build Table and Copy to Clipboard
# ---------------------------------------------------------
print("Building summary table...")
records = []

for feature in features:
    # Build a row containing the feature name, its 5 fold values, and the overall average
    row = {"Feature": feature}
    for i in range(5):
        row[f"Fold_{i+1}_SHAP"] = shap_results[feature][i]
        
    row["Overall_Average_SHAP"] = np.mean(shap_results[feature])
    records.append(row)

# Convert to DataFrame
results_df = pd.DataFrame(records)

# Sort from most important to least important (based on overall average)
results_df = results_df.sort_values(by="Overall_Average_SHAP", ascending=False).reset_index(drop=True)

# Print cleanly to the console
print("\n" + "="*85)
print(" SHAP CROSS-VALIDATION SENSITIVITY TABLE (REGRESSION) ".center(85, "="))
print("="*85)
print(results_df.to_string(index=False))

# Save directly to clipboard
results_df.to_clipboard(index=False)
print("\n✅ Valid SHAP Cross-Validation table copied to clipboard! You can paste it directly into Excel.")