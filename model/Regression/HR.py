import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor  # Swapped from GBR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load reordered data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_HR" 

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]

X = df.drop(columns=target_column)
y = df[target_column]

# --- Use last 20% as test set ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Define and train Huber Regression model ---
    # "epsilon": 2.78677,
    # "max_iter": 30,
    # "alpha": 0.0760023933,
# We use a pipeline to scale data before feeding it into the Huber model
model = HuberRegressor(
        epsilon=2.78677,
        max_iter=27,
        alpha=0.0760023933,
        # fit_intercept=True
    )

model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Updated Metrics Function ---
def compute_metrics(y_true, y_hat):
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    mae = mean_absolute_error(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)
    
    # COV: (RMSE / Mean of True Values) * 100
    mean_actual = np.mean(y_true)
    cov = (rmse / mean_actual * 100) if mean_actual != 0 else 0.0
    
    # U95: 1.96 * Standard Deviation of Residuals
    u95 = 1.96 * np.std(y_true - y_hat)
    
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "COV": cov, "U95": u95}

# --- Build Metrics Table ---
mid = len(y_test) // 2
set_data = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test[:mid], y_pred_test[:mid]),
    ("Value-test", y_test[mid:], y_pred_test[mid:]),
]

metrics_list = []
for name, y_t, y_p in set_data:
    m = compute_metrics(y_t, y_p)
    m["Set"] = name
    metrics_list.append(m)

df_metrics = pd.DataFrame(metrics_list)[["Set", "R2", "RMSE", "MAE", "COV", "U95"]]
print(df_metrics)

# --- Export to clipboard ---
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
df_all.to_clipboard(index=False, header=False)