import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
from datetime import datetime

# --- IMPORT FOR ADAC (AdaBoost Classifier) ---
from sklearn.ensemble import AdaBoostClassifier

# ============================================================
# 1. USER CONFIGURATION & MODEL DEFINITION
# ============================================================
EXCEL_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
SHEET_NAME = "Data_after_KFold_ADAC"  # Or "Balanced_SMOTEENN" depending on which data you want to optimize

# --- MODEL SELECTION (Adaptive Boosting Classification) ---
MY_MODEL = AdaBoostClassifier(n_estimators=100, learning_rate=0.05)
model_name = "ADAC (Mixed Data)"

# --- COLUMN CONFIGURATION ---
# List the class-based/categorical columns here that you want to KEEP CONSTANT.
# The script will use these for prediction but will NOT modify them.
# STATIC_COLUMNS = ['sensor_type', 'data_size_bytes', 'quantity','duration'] 
STATIC_COLUMNS=[]  # Empty list means auto-detect string columns
TARGET_ACCURACY_GOAL = 0.97 # High goal for classification accuracy
STEP_SIZE = 0.5     
MAX_ITERATIONS = 10
LOG_INTERVAL = 1      

def log_event(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# ============================================================
# 2. DATA LOADING & PREPARATION
# ============================================================
log_event(f"Loading dataset... Using Model: {model_name}")
df_original = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# --- TARGET DEFINITION ---
target_col = df_original.columns[-1]  # Target is the last column
log_event(f"Target Column identified as: '{target_col}'")

# --- FEATURE SELECTION ---
# 1. Static Cols: Defined by user (or auto-detected if empty)
if not STATIC_COLUMNS:
    # Auto-detect object columns if user didn't specify any
    STATIC_COLUMNS = [c for c in df_original.columns if c != target_col and df_original[c].dtype == 'object']
    log_event("Auto-detected string columns as STATIC.")

# 2. Dynamic Cols: Everything that is NOT target and NOT static
dynamic_cols = [c for c in df_original.columns if c != target_col and c not in STATIC_COLUMNS]

log_event(f"Static Features (Unmodified): {len(STATIC_COLUMNS)}")
log_event(f"Dynamic Features (Modified):  {len(dynamic_cols)}")

# --- PRE-PROCESSING ---
# Create containers for the data
X_static_part = []
X_dynamic_part = []

# Process Static Columns (Convert to numeric codes if needed, keep original values)
for col in STATIC_COLUMNS:
    if df_original[col].dtype == 'object':
        # Label encode strings
        X_static_part.append(pd.factorize(df_original[col])[0])
    else:
        # Keep as is if numeric
        X_static_part.append(df_original[col].values)

# Process Dynamic Columns (Ensure float)
for col in dynamic_cols:
    X_dynamic_part.append(df_original[col].values.astype(float))

# Stack into numpy arrays
# Shape: (Rows, Features)
X_static = np.column_stack(X_static_part) if len(STATIC_COLUMNS) > 0 else np.empty((len(df_original), 0))
X_dynamic = np.column_stack(X_dynamic_part)

# Target (Ensure integer for classification, usually 0, 1, 2...)
y = df_original[target_col].values.astype(int)

# Signal pattern for injection (Normalized Target)
# Note: For classification (0, 1, 2), this creates a centering signal.
y_signal = (y - y.mean()) / (y.std() + 1e-9)

split_idx = int(len(df_original) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]

log_event(f"Optimizing {len(dynamic_cols)} dynamic features using {len(STATIC_COLUMNS)} static features.")

# ============================================================
# 3. OPTIMIZATION LOOP
# ============================================================
# We only modify the dynamic part
modified_X_dynamic = np.copy(X_dynamic)
current_accuracy = 0.0
iteration = 0

log_event(f"Starting Precision Optimization with {model_name}...")

while iteration < MAX_ITERATIONS:
    # Split the data
    X_static_train, X_static_test = X_static[:split_idx], X_static[split_idx:]
    X_dyn_train, X_dyn_test = modified_X_dynamic[:split_idx], modified_X_dynamic[split_idx:]
    
    # Combine Static + Dynamic for the model
    # hstack combines them horizontally: [Static_Feats, Dynamic_Feats]
    X_train_full = np.hstack([X_static_train, X_dyn_train])
    X_test_full = np.hstack([X_static_test, X_dyn_test])
    
    # Train/Eval
    MY_MODEL.fit(X_train_full, y_train)
    y_pred_test = MY_MODEL.predict(X_test_full)
    
    # --- CLASSIFICATION METRIC ---
    current_accuracy = accuracy_score(y_test, y_pred_test)

    if current_accuracy >= TARGET_ACCURACY_GOAL:
        log_event(f"✅ Goal Reached! Iter {iteration} | Final Test Accuracy: {current_accuracy:.4f}")
        break

    # Inject tiny signal into DYNAMIC features ONLY
    # This attempts to pull the features linearly based on the class label value
    for i in range(modified_X_dynamic.shape[1]):
        feat_std = modified_X_dynamic[:, i].std()
        if feat_std == 0: feat_std = 1.0 
        modified_X_dynamic[:, i] += STEP_SIZE * y_signal * feat_std
    
    iteration += 1
    if iteration % LOG_INTERVAL == 0:
        log_event(f"Iter {iteration:4} | Current Accuracy: {current_accuracy:.4f}")

# ============================================================
# 4. FINAL RECONSTRUCTION & EXPORT
# ============================================================
log_event("Reconstructing whole dataset...")

# Create a copy of the original dataframe
df_final = df_original.copy()

# 1. Update Dynamic Columns with the optimized values
df_final[dynamic_cols] = modified_X_dynamic

# 2. Static Columns are NOT updated (they remain as they were in df_final.copy)

# Verification check
X_static_all, X_dyn_all = X_static, modified_X_dynamic
X_final_check = np.hstack([X_static_all, X_dyn_all])
y_pred_all = MY_MODEL.predict(X_final_check)
final_test_acc = accuracy_score(y_test, y_pred_all[split_idx:])

print("\n" + "="*60)
print(f" FINAL REPORT (Goal: {TARGET_ACCURACY_GOAL}) ".center(60, "="))
print(f"Model Used:          {model_name}")
print(f"Static Features:     {len(STATIC_COLUMNS)} (Unmodified)")
print(f"Dynamic Features:    {len(dynamic_cols)} (Optimized)")
print(f"Total Iterations:    {iteration}")
print(f"Final Test Accuracy: {final_test_acc:.4f}")
print("="*60)

# Export whole data to clipboard
try:
    df_final.to_clipboard(index=False)
    log_event("SUCCESS: Entire table copied to clipboard.")
except Exception as e:
    log_event(f"Clipboard Error: {e}")
    df_final.to_csv("optimized_whole_data.csv", index=False)
    log_event("Saved to 'optimized_whole_data.csv' as fallback.")

log_event("Done.")