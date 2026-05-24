import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# Load Excel data
# =========================
df = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    sheet_name="Wilcoxon test Data",
    header=0,
)

# =========================
# Extract models dynamically
# =========================
columns = df.columns.tolist()
structured_data = []

for i in range(0, len(columns), 2):

    name = columns[i].strip()

    y_real = df.iloc[:, i].to_numpy()
    y_predict = df.iloc[:, i + 1].to_numpy()

    structured_data.append({
        "name": name,
        "y_real": y_real,
        "y_predict": y_predict
    })

# =========================
# Bootstrap CI Function
# =========================
def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_func,
    n_bootstrap=2000,
    ci=95,
    random_state=42
):

    rng = np.random.default_rng(random_state)

    scores = []

    n = len(y_true)

    for _ in range(n_bootstrap):

        # resample indices with replacement
        indices = rng.choice(n, n, replace=True)

        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        score = metric_func(y_true_sample, y_pred_sample)

        scores.append(score)

    scores = np.array(scores)

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)

    return (
        np.mean(scores),
        lower,
        upper
    )

# =========================
# Metrics
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

metrics = {
    "MAE": mean_absolute_error,
    "RMSE": rmse,
    "R2": r2_score,
}

# =========================
# Calculate CI for all models
# =========================
results = []

for entry in structured_data:

    model_name = entry["name"]

    y_true = entry["y_real"]
    y_pred = entry["y_predict"]

    for metric_name, metric_func in metrics.items():

        mean_score, ci_lower, ci_upper = bootstrap_metric_ci(
            y_true,
            y_pred,
            metric_func,
            n_bootstrap=2000,
            ci=95
        )

        results.append({
            "Model": model_name,
            "Metric": metric_name,
            "Mean": mean_score,
            "CI Lower": ci_lower,
            "CI Upper": ci_upper,
        })

# =========================
# Final DataFrame
# =========================
df_results = pd.DataFrame(results)

print(df_results)

# copy to clipboard
df_results.to_clipboard(index=False)