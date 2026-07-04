import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load reordered data for ENR (after K-Fold)
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "KfoldETR"


df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]


# Use last 20% as test set to match K-Fold logic
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train and predict
model = ElasticNet(
    alpha=0.2,
    l1_ratio=0.5,
    max_iter=1000,
    tol=2e-4,
    # random_state=42
)
model.fit(X_train, y_train)
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
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
df_all.to_clipboard(index=False,header=False)
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)