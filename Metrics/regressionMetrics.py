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

    # Avoid division by zero
    nonzero_mask = np.abs(y_true) > 1e-8
    rel_error = np.zeros_like(y_true)
    rel_error[nonzero_mask] = abs_error[nonzero_mask] / np.abs(y_true[nonzero_mask])

    # Mean Absolute Relative Error (MARE)
    mare = np.mean(rel_error) * 100

    # RMSE
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    # R2
    r2 = r2_score(y_true, y_pred)

    # 95th percentile of absolute error (U95)
    u95 = np.percentile(abs_error, 95)

    # Minimum Sum of Absolute Error (MSAE)
    # (Used as an optimization-oriented metric)
    msae = np.min(np.cumsum(abs_error))

    return {
        "R2": r2,
        "RMSE": rmse,
        "MARE": mare,
        "U95": u95,
        "MSAE": msae
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
    columns=["Set", "R2", "RMSE", "MARE", "U95", "MSAE"],
)

# -------------------- 6. Output --------------------
print(df_main)
df_main.to_clipboard(index=False)
