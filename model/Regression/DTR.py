import pandas as pd
import numpy as np
import win32com.client
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Sheet1"

# Load the dataframe
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# 🚩 AUTOMATIC LEAK DETECTION
# Before we split, let's find any column that is unique for every row (acting like an ID)
unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
if unique_cols:
    print(f"⚠️ FOUND UNIQUE COLUMNS (LEAKAGE RISK): {unique_cols}")

# --- PREPARE DATA ---
target_column = df.columns[-1]

# 1. Drop the target
# 2. Drop the specific leaky column you found ('energy_consumption_kWh')
# 3. Drop any other unique columns found automatically
# cols_to_drop = [target_column, 'energy_consumption_kWh'] + unique_cols
X = df.drop(columns=list(set(cols_to_drop))) # set() avoids double-dropping
y = df[target_column]

# --- Train/Test Split (80/20) ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- PRUNED MODEL ---
# We use max_depth and min_samples_leaf to force the model to GENERALIZE
model = DecisionTreeRegressor(
    max_depth=4,           # Stop the tree from growing too deep
    min_samples_leaf=10,    # Each prediction must be based on at least 10 samples
    random_state=42
)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Metrics Calculation ---
mid = len(y_test) // 2
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test[:mid], y_pred_test[:mid]),
    ("Test-Value", y_test[mid:], y_pred_test[mid:])
]

df_metrics = pd.DataFrame([{
    "Set": s,
    "MAE": mean_absolute_error(y_t, y_p),
    "RMSE": mean_squared_error(y_t, y_p) ** 0.5,
    "R2": r2_score(y_t, y_p)
} for s, y_t, y_p in sets])

# --- RESULTS ---
print("\n--- Model Performance (Pruned & Cleaned) ---")
print(df_metrics)

print("\n--- Correlation check (signals vs noise) ---")
correlations = df.corr()[target_column].sort_values(ascending=False)
print(correlations.head(6))

# --- Export predictions ---
# Exporting 'All' predictions to clipboard for easy Excel pasting
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
df_all.to_clipboard(index=False, header=False)
print("\n✅ Cleaned predictions copied to clipboard.")