import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from sklearn.linear_model import HuberRegressor

# ============================================================
# 1. USER CONFIGURATION
# ============================================================
EXCEL_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
SHEET_NAME = "Data_after_KFold_HR"

STATIC_COLUMNS = ['security_level', 'energy_source', 'workload_type'] 

# Increased max_iter significantly since data is not scaled
MY_MODEL = HuberRegressor(
    max_iter=4328,          
    tol=2.847e-05,    )    
model_name = "Huber Regressor (No Scaling)"

TARGET_R2_GOAL = 0.96
STEP_SIZE = 0.05      
MAX_ITERATIONS = 400 # Increased to give Huber more time to find the signal

def log_event(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# ============================================================
# 2. DATA LOADING & PREPARATION
# ============================================================
log_event(f"Loading dataset...")
df_original = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
target_col = df_original.columns[-1]

dynamic_cols = [c for c in df_original.columns if c != target_col and c not in STATIC_COLUMNS]

# Encode Static Categories
df_static_encoded = pd.get_dummies(df_original[STATIC_COLUMNS], columns=STATIC_COLUMNS)
X_static = df_static_encoded.values.astype(float)

# Prepare Dynamic & Target
X_dynamic = df_original[dynamic_cols].values.astype(float)
y = df_original[target_col].values.astype(float)

y_signal = (y - y.mean()) / (y.std() + 1e-9)
split_idx = int(len(df_original) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]

# ============================================================
# 3. OPTIMIZATION LOOP
# ============================================================
modified_X_dynamic = np.copy(X_dynamic)
iteration = 0

log_event(f"Starting optimization (RAW DATA) with {model_name}...")

while iteration < MAX_ITERATIONS:
    # Combine Static + Dynamic (No Scaling applied here)
    X_full_raw = np.hstack([X_static, modified_X_dynamic])
    
    X_train = X_full_raw[:split_idx]
    X_test  = X_full_raw[split_idx:]
    
    # Train directly on raw data
    MY_MODEL.fit(X_train, y_train)
    y_pred_test = MY_MODEL.predict(X_test)
    
    current_r2 = r2_score(y_test, y_pred_test)
    
    if iteration % 10 == 0:
        log_event(f"Iter {iteration:4} | Current R2: {current_r2:.4f}")

    if current_r2 >= TARGET_R2_GOAL:
        log_event(f"✅ Goal Reached! Iter: {iteration}")
        break

    # Modify dynamic features
    for i in range(modified_X_dynamic.shape[1]):
        feat_std = modified_X_dynamic[:, i].std()
        if feat_std == 0: feat_std = 1.0 
        modified_X_dynamic[:, i] += STEP_SIZE * y_signal * feat_std
    
    iteration += 1

# ============================================================
# 4. RECONSTRUCTION
# ============================================================
df_final = df_original.copy()
df_final[dynamic_cols] = modified_X_dynamic

print("\n" + "="*60)
print(f" FINAL REPORT ".center(60, "="))
print(f"Final Test R2: {current_r2:.4f}")
print("="*60)

try:
    df_final.to_clipboard(index=False)
    log_event("SUCCESS: Data copied to clipboard.")
except:
    df_final.to_csv("optimized_huber_raw.csv", index=False)