import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# -------------------- 1. Load data --------------------
data = np.loadtxt(r"C:\Users\Sam\Desktop\ML\data\Data_err.npt")
y_real = data[:, 0]
y_pred = data[:, 1]

# -------------------- 2. Split into train/test --------------------
split_idx = int(len(y_real) * 0.8)
y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
y_pred_train = y_pred[:split_idx]
y_pred_test = y_pred[split_idx:]

# -------------------- 3. Define regression metrics --------------------
def get_regression_metrics(y_true, y_pred):
    abs_error = np.abs(y_true - y_pred)
    rel_error = abs_error / np.abs(y_true)

    # Prediction Interval (PI): % of predictions within ±10% of true value
    within_10_percent = rel_error <= 0.10
    pi = np.mean(within_10_percent) * 100

    # Coefficient of Variation (COV): std / mean of predictions
    cov = np.std(y_pred) / np.mean(y_pred)

    # Mean Absolute Relative Deviation (MARD)
    mard = np.mean(rel_error) * 100

    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "PI": pi,
        "COV": cov,
        "MARD": mard
    }

# -------------------- 4. Compute metrics --------------------
metrics_all = get_regression_metrics(y_real, y_pred)
metrics_train = get_regression_metrics(y_real_train, y_pred_train)
metrics_test = get_regression_metrics(y_real_test, y_pred_test)

# Split test into Value and Value-test
mid = len(y_real_test) // 2
y_real_value, y_pred_value = y_real_test[:mid], y_pred_test[:mid]
y_real_value_test, y_pred_value_test = y_real_test[mid:], y_pred_test[mid:]

metrics_value = get_regression_metrics(y_real_value, y_pred_value)
metrics_value_test = get_regression_metrics(y_real_value_test, y_pred_value_test)

# -------------------- 5. Create metrics DataFrame --------------------
df_main = pd.DataFrame(
    [
        ["All", *metrics_all.values()],
        ["Train", *metrics_train.values()],
        ["Test", *metrics_test.values()],
        ["Value", *metrics_value.values()],
        ["Value-test", *metrics_value_test.values()],
    ],
    columns=["Set", "R2", "RMSE", "PI", "COV", "MARD"],
)

# -------------------- 6. Save to clipboard --------------------
print(df_main)
df_main.to_clipboard(index=False)