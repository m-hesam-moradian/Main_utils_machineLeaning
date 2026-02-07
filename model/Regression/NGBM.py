import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# --- NGBOOST IMPORT ---
from ngboost import NGBRegressor

# --- Load reordered data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_NGBM" 

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]

X = df.drop(columns=target_column)
y = df[target_column]

# --- Random Train/Test Split (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       
    random_state=42,     
    shuffle=True         
)

# --- 1. CLEAN DATA ---
X_train_np = X_train.to_numpy(dtype=np.float64)
y_train_np = y_train.to_numpy(dtype=np.float64).flatten()
X_test_np = X_test.to_numpy(dtype=np.float64)
X_all_np = X.to_numpy(dtype=np.float64)

# --- 2. THE CRITICAL FIX: natural_gradient=False ---
# This prevents the 'ValueError: solve' mismatch by using standard gradients
model = NGBRegressor(
n_estimators=312,           # High tree count to capture complex patterns
    learning_rate=0.00947,
    natural_gradient=False,     # <--- DISABLING THIS FIXES THE CRASH
    
    random_state=42
)

# --- 3. Train ---
print("Training NGBM (Standard Gradient mode)...")
model.fit(X_train_np, y_train_np)

# --- 4. Predictions ---
y_pred_all = model.predict(X_all_np)
y_pred_train = model.predict(X_train_np)
y_pred_test = model.predict(X_test_np)

# --- 5. Metrics ---
mid = len(y_test) // 2
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test[:mid], y_pred_test[:mid]),
    ("Test-Value", y_test[mid:], y_pred_test[mid:]),
]

df_metrics = pd.DataFrame(
    [
        {
            "Set": s,
            "MAE": mean_absolute_error(y_t, y_p),
            "RMSE": mean_squared_error(y_t, y_p) ** 0.5,
            "R2": r2_score(y_t, y_p),
        }
        for s, y_t, y_p in sets
    ]
)

# --- Display Results ---
print("\n" + "="*50)
print(" NGBoost Regression Summary (Crash-Fix Mode) ".center(50, "="))
print(df_metrics)
print("="*50)

# --- Export to clipboard ---
try:
    df_result = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
    df_result.to_clipboard(index=False, header=False)
    print("\nSUCCESS: Predictions copied to clipboard.")
except Exception as e:
    print(f"\nClipboard Error: {e}")