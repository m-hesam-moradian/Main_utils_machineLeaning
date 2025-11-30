import numpy as np
import pandas as pd

# 1. Load data
data = np.loadtxt(r"C:\Users\Sam\Desktop\ML\data\Data_err.npt")
y_real = data[:, 0]
y_pred = data[:, 1]

# 2. Split into train/test
split_idx = int(len(y_real) * 0.8)
y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
y_pred_train = y_pred[:split_idx]
y_pred_test = y_pred[split_idx:]

# 3. Define regression metrics
def get_regression_metrics(y_true, y_pred):
    abs_error = np.abs(y_true - y_pred)
    nonzero_mask = np.abs(y_true) > 1e-8
    rel_error = np.zeros_like(y_true)
    rel_error[nonzero_mask] = abs_error[nonzero_mask] / np.abs(y_true[nonzero_mask])
    gr100 = np.mean(rel_error <= 1.0) * 100
    gr125 = np.mean(rel_error <= 1.25) * 100
    return {"GR100": gr100, "GR125": gr125}

# 4. Compute metrics
metrics_all   = get_regression_metrics(y_real, y_pred)
metrics_train = get_regression_metrics(y_real_train, y_pred_train)
metrics_test  = get_regression_metrics(y_real_test, y_pred_test)

mid = len(y_real_test) // 2
y_real_value, y_pred_value = y_real_test[:mid], y_pred_test[:mid]
y_real_value_test, y_pred_value_test = y_real_test[mid:], y_pred_test[mid:]
metrics_value      = get_regression_metrics(y_real_value, y_pred_value)
metrics_value_test = get_regression_metrics(y_real_value_test, y_pred_value_test)

# 5. Create metrics DataFrame
df_main = pd.DataFrame(
    [
        ["All",        metrics_all["GR100"],   metrics_all["GR125"]],
        ["Train",      metrics_train["GR100"], metrics_train["GR125"]],
        ["Test",       metrics_test["GR100"],  metrics_test["GR125"]],
        ["Value",      metrics_value["GR100"], metrics_value["GR125"]],
        ["Value-test", metrics_value_test["GR100"], metrics_value_test["GR125"]],
    ],
    columns=["Set", "GR100", "GR125"],
)

# 6. Save to clipboard
print(df_main)
df_main.to_clipboard(index=False)