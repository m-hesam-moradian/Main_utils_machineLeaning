import pandas as pd
import numpy as np

# ============================
# Paths
# ============================
input_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
output_path = r"C:\Users\Sam\Desktop\ML\task\Quantile_Loss_Results.xlsx"

# ============================
# Settings
# ============================
TAU = 0.5
N_BOOTSTRAPS = 1000
CONFIDENCE = 95
RANDOM_STATE = 42

# ============================
# Load data
# ============================
df = pd.read_excel(
    input_path,
    header=0,
    sheet_name="predicts(VIF)"
)

columns = df.columns.tolist()

results = []

rng = np.random.default_rng(RANDOM_STATE)

# ============================
# Quantile Loss
# ============================
def quantile_loss(y_true, y_pred, tau=0.5):
    error = y_true - y_pred
    loss = np.maximum(tau * error, (tau - 1) * error)
    return np.mean(loss)

# ============================
# Bootstrap Confidence Interval
# ============================
def bootstrap_ci(y_true, y_pred,
                 tau=0.5,
                 n_bootstraps=1000,
                 confidence=95):

    n = len(y_true)

    scores = np.empty(n_bootstraps)

    for i in range(n_bootstraps):
        idx = rng.integers(0, n, n)

        scores[i] = quantile_loss(
            y_true[idx],
            y_pred[idx],
            tau
        )

    alpha = (100 - confidence) / 2

    lower = np.percentile(scores, alpha)
    upper = np.percentile(scores, 100 - alpha)

    return lower, upper

# ============================
# Calculate Metrics
# ============================
for i in range(0, len(columns), 2):

    model_name = columns[i].strip()

    temp_df = df.iloc[:, [i, i + 1]].dropna()

    y_true = temp_df.iloc[:, 0].to_numpy(dtype=float)
    y_pred = temp_df.iloc[:, 1].to_numpy(dtype=float)

    qloss = quantile_loss(y_true, y_pred, TAU)

    ci_low, ci_high = bootstrap_ci(
        y_true,
        y_pred,
        tau=TAU,
        n_bootstraps=N_BOOTSTRAPS,
        confidence=CONFIDENCE
    )

    results.append({
        "Model": model_name,
        "Quantile": TAU,
        "Quantile Loss": qloss,
        "CI Lower": ci_low,
        "CI Upper": ci_high,
        "95% CI": f"[{ci_low:.6f}, {ci_high:.6f}]"
    })

# ============================
# Save Results
# ============================
result_df = pd.DataFrame(results)

result_df.to_excel(output_path, index=False)

print(f"✅ Results saved to:\n{output_path}")

try:
    result_df.to_clipboard(index=False)
    print("📋 Results copied to clipboard.")
except Exception:
    pass

print("\nPreview:\n")
print(result_df.to_string(index=False))