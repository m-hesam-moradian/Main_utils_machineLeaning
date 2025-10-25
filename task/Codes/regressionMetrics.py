import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------- 1. Load data --------------------
data = np.loadtxt(r"C:\Users\Sam\Desktop\ML\data\Data_err.npt")
y_real = data[:, 0]
y_pred = data[:, 1]

# -------------------- 2. Split into train/test --------------------
split_idx = int(len(y_real) * 0.8)
y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
y_pred_train, y_pred_test = y_pred[:split_idx], y_pred[split_idx:]

# -------------------- 3. Define regression metrics --------------------
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    return np.mean(diff / denominator) * 100

def gr_threshold(y_true, y_pred, threshold, percent=False):
    error = np.abs(y_true - y_pred)
    if percent:
        threshold_values = np.abs(y_true) * (threshold / 100)
        return np.mean(error <= threshold_values) * 100
    else:
        return np.mean(error <= threshold) * 100
def get_regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "GR10": gr_threshold(y_true, y_pred, 10),               # absolute error ≤ 10
        "GR10%": gr_threshold(y_true, y_pred, 10, percent=True) # error ≤ 10% of true value
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
    columns=["Set", "R2", "RMSE", "MAE", "SMAPE", "GR100", "GR125"],
)

# -------------------- 6. Save to clipboard --------------------
print(df_main)
df_main.to_clipboard(index=False)