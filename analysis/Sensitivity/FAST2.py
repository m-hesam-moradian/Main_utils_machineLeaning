import pandas as pd
import numpy as np
import time
from SALib.sample import fast_sampler
from SALib.analyze import fast
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor # <-- Changed to Regressors!
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Load Original Dataset & Train Model
# ---------------------------------------------------------
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="KfoldXGBR").dropna()
target_column = df.columns[-1]

# Separate features and target
X_real = df.drop(columns=[target_column])
y_real = df[target_column]

print("Training ML model to evaluate FAST samples...")
# Changed to RandomForestRegressor for continuous target variables
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_real, y_real)

# ---------------------------------------------------------
# 2. Define the FAST Problem
# ---------------------------------------------------------
D = X_real.shape[1]
problem = {
    "num_vars": D,
    "names": list(X_real.columns),
    "bounds": [[X_real[col].min(), X_real[col].max()] for col in X_real.columns]
}

# ---------------------------------------------------------
# 3. Generate Mathematical FAST Samples
# ---------------------------------------------------------
print("Generating mathematical FAST samples...")
N_samples = 20000 
X_fast_matrix = fast_sampler.sample(problem, N_samples)
X_fast_df = pd.DataFrame(X_fast_matrix, columns=X_real.columns)

# ---------------------------------------------------------
# 4. Predict using the Model
# ---------------------------------------------------------
print("Predicting outcomes for FAST samples...")
# REGRESSION FIX: No more TARGET_CLASS or predict_proba. 
# We just predict the continuous output directly!
y_fast_predictions = model.predict(X_fast_df)

# ---------------------------------------------------------
# 5. Run FAST Analysis
# ---------------------------------------------------------
print("Running FAST analysis for Regression...")
start_time = time.time()
Si = fast.analyze(problem, y_fast_predictions, print_to_console=False)
end_time = time.time()

# ---------------------------------------------------------
# 6. Build Results DataFrame
# ---------------------------------------------------------
Fast_df = pd.DataFrame({
    "parameter": problem["names"],
    "S1": Si["S1"],
    "S1_conf": Si["S1_conf"],
    "ST": Si["ST"],
    "ST_conf": Si["ST_conf"]
})

# Keep precision up to 10 decimal places instead of rounding to 4
Fast_df["S1"] = Fast_df["S1"].clip(lower=0.0).round(10)
Fast_df["ST"] = Fast_df["ST"].clip(lower=0.0).round(10)
Fast_df["S1_conf"] = Fast_df["S1_conf"].round(10)
Fast_df["ST_conf"] = Fast_df["ST_conf"].round(10)

# Sort by First-Order Sensitivity (S1)
Fast_df = Fast_df.sort_values(by="S1", ascending=False)

print(f"FAST analysis done in {end_time - start_time:.2f} seconds.")
print("\n=== FAST Sensitivity Results (Regression) ===")

# Force pandas to print out all 10 decimal places in the console
pd.set_option('display.float_format', lambda x: '%.10f' % x)
print(Fast_df)

# Copy to clipboard
Fast_df.to_clipboard(index=False)
print("\n✅ High-precision FAST results copied to clipboard!")