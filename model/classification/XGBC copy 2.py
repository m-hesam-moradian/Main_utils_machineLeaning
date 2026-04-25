import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# --- Load data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Encoded_Data_Sheet2"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]

# --- Prepare features and target ---
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Train/Test split ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Model ---
model = XGBRegressor(
    # n_estimators=300,
    # max_depth=1,
    # learning_rate=0.1
)

# --- Train ---
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ✅ --- OPTIONAL: Round predictions (since your values are integers) ---
y_pred_all = np.round(y_pred_all)
y_pred_train = np.round(y_pred_train)
y_pred_test = np.round(y_pred_test)

# --- Metrics ---
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
    "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
    "R2": r2_score(y_t, y_p)
} for s, y_t, y_p in sets])

print(df_metrics)

# --- Output ---
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# --- Export ---
df_all.to_clipboard(index=False, header=False)
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)