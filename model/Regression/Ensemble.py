import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load reordered data ---
# Note: Ensure the path matches your local machine exactly
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_HGBR"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# --- Use last 20% as test set to match K-Fold logic ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Define the Ensemble ---

# 1. Define Decision Tree
dt_model = HistGradientBoostingRegressor(
    random_state=42,
    max_iter=250,
    learning_rate=0.05
)


# 2. Define HistGradientBoostingRegressor
hgbr_model = HistGradientBoostingRegressor(
    random_state=42,
    max_iter=1200,
    learning_rate=0.05
)

# 3. Create the Voting Regressor (Ensemble)
# This will average the predictions of both models
model = VotingRegressor(estimators=[
    ('hgbr1', dt_model),
    ('hgbr2', hgbr_model)
])

# --- Train and predict ---
model.fit(X_train, y_train)

y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

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
    "RMSE": mean_squared_error(y_t, y_p) ** 0.5,
    "R2": r2_score(y_t, y_p)
} for s, y_t, y_p in sets])

print(df_metrics)

# --- Output predictions ---
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# --- Export to clipboard ---
# Be careful: writing multiple dataframes to clipboard sequentially 
# will overwrite the previous ones. The script below leaves df_test in the clipboard.
df_all.to_clipboard(index=False, header=False)
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)