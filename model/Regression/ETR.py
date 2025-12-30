import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load reordered data for ETR (after K-Fold)
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "data_after_vif"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# Use last 20% as test set to match K-Fold logic
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train and predict

model = ExtraTreesRegressor(
    n_estimators=100,         # 500 is overkill and increases memorization
    max_depth=5,              # LOWER THIS significantly (try 3, 4, or 5)
    min_samples_split=10,      # Force it to look at at least 10 rows before splitting
    min_samples_leaf=5,       # Each leaf MUST represent at least 5 different rows
    max_features=1.0,         # Try using all features to see if there is any real signal
    random_state=42
)

model.fit(X_train, y_train)
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train) 
y_pred_test = model.predict(X_test)


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

print(df_metrics)

# Output predictions
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# Optional: Export to clipboard or Excel
df_all.to_clipboard()
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)