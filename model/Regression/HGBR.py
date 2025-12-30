import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================== Load Data ==================
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_RR"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

target_column = df.columns[-1]
X_full = df.drop(columns=[target_column])
y = df[target_column]

# Drop ONLY for model input (same as K-Fold)
COLUMNS_TO_DROP = ['workload_type', 'energy_source', 'security_level', 'pqc_enabled']
X_model = X_full.drop(columns=COLUMNS_TO_DROP, errors="ignore")

# ================== Train / Test Split (80/20) ==================
split_idx = int(len(df) * 0.8)

X_train = X_model.iloc[:split_idx]
X_test  = X_model.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

# ================== Train Model ==================
model = Ridge(
    random_state=42,
    alpha=1.0,
    max_iter=10
)

model.fit(X_train, y_train)

# ================== Predictions ==================
y_pred_all = model.predict(X_model)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ================== Metrics ==================
mid = len(y_test) // 2

sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test.iloc[:mid], y_pred_test[:mid]),
    ("Test-Value", y_test.iloc[mid:], y_pred_test[mid:])
]

df_metrics = pd.DataFrame([
    {
        "Set": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
    for name, y_true, y_pred in sets
])

print(df_metrics)

# ================== Prediction Outputs ==================
df_all = pd.DataFrame({
    "y_real": y,
    "y_pred": y_pred_all
})

df_train = pd.DataFrame({
    "y_real": y_train,
    "y_pred": y_pred_train
})

df_test = pd.DataFrame({
    "y_real": y_test,
    "y_pred": y_pred_test
})

# ================== Export ==================
df_all.to_clipboard(index=False, header=False)
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)
