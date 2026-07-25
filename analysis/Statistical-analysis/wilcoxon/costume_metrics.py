import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

EXCEL_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
SHEET_NAME = "predicts"


def compute_metrics(y_true, y_pred, delta=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_pred - y_true
    non_zero = y_true != 0
    mean_y = np.mean(y_true)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MARE and MAPE aligned with standard table definitions
    mare = (
        np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))
        if np.any(non_zero)
        else np.nan
    )
    mape = mare  # Kept identical as per provided metric criteria

    si = rmse / mean_y if mean_y != 0 else np.nan
    cov = si  # Coefficient of Variation defined relative to mean

    # U95 calculation based on standard deviation of residuals
    sd = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    u95 = 1.96 * np.sqrt(sd**2 + (np.mean(diff)) ** 2)

    mbe = np.mean(diff)

    # Log-Cosh Loss
    logcosh = np.mean(
        np.abs(diff) - np.log(2.0) + np.log1p(np.exp(-2.0 * np.abs(diff)))
    )

    # Huber Loss
    is_small = np.abs(diff) <= delta
    huber = np.mean(
        np.where(is_small, 0.5 * diff**2, delta * np.abs(diff) - 0.5 * delta**2)
    )

    com = -mbe / mean_y if mean_y != 0 else np.nan

    denom = np.sum(np.abs(y_true - mean_y))
    rae = np.sum(np.abs(diff)) / denom if denom != 0 else np.nan

    return {
        "R2": r2,
        "RMSE": rmse,
        "MARE": mare,
        "COV": cov,
        "U95": u95,
        "SI": si,
        "MBE": mbe,
        "LogCosh Loss": logcosh,
        "Huber Loss": huber,
        "MAPE": mape,
        "COM": com,
        "RAE": rae,
    }


def split_data(y_real, y_pred):
    n = len(y_real)
    n_train = int(n * 0.70)
    n_val = int(n * 0.10)

    splits = {
        "Train": (y_real[:n_train], y_pred[:n_train]),
        "Val.": (y_real[n_train : n_train + n_val], y_pred[n_train : n_train + n_val]),
        "Test": (y_real[n_train + n_val :], y_pred[n_train + n_val :]),
    }
    return splits


# Load data
df = pd.read_excel(EXCEL_PATH, header=0, sheet_name=SHEET_NAME)
columns = df.columns.tolist()

# Process metrics per model and split
model_results = {}

for i in range(0, len(columns), 2):
    model_name = columns[i].strip()
    y_real = df.iloc[:, i].to_numpy(dtype=float)
    y_pred = df.iloc[:, i + 1].to_numpy(dtype=float)

    splits = split_data(y_real, y_pred)
    model_results[model_name] = {}

    for split_name, (r_split, p_split) in splits.items():
        model_results[model_name][split_name] = compute_metrics(r_split, p_split)

# Convert to MultiIndex DataFrame
formatted_data = {}
for model, splits in model_results.items():
    for split_name, metrics in splits.items():
        for metric_name, val in metrics.items():
            if metric_name not in formatted_data:
                formatted_data[metric_name] = {}
            formatted_data[metric_name][(model, split_name)] = val

table = pd.DataFrame(formatted_data).T

# Order columns as (Model, Split)
table.columns = pd.MultiIndex.from_tuples(
    table.columns, names=["Model", "Partition"]
)

# Display and copy results
print(table.round(3))
table.to_clipboard()