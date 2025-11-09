# =========================================================
# uncertainty_mcs_table7_from_predictions_fixed.py
# =========================================================
import numpy as np
import pandas as pd

# -------------------------
# USER SETTINGS
DATA_PATH = r"C:\Users\Sam\Desktop\ML\data\Data_err.npt"  # .npt file with [actual, predicted]
MODEL_NAME = "HGBR"  # e.g., "ENR", "XGBR", etc.
# -------------------------

# Load data
data = np.loadtxt(DATA_PATH)
CSac = data[:, 0]  # actual
CSpr = data[:, 1]  # predicted

# Convert to float arrays and clean NaN / inf / negatives
CSac = np.asarray(CSac, dtype=float)
CSpr = np.asarray(CSpr, dtype=float)

mask_valid = np.isfinite(CSac) & np.isfinite(CSpr) & (CSac > 0) & (CSpr > 0)
CSac = CSac[mask_valid]
CSpr = CSpr[mask_valid]

if len(CSac) == 0:
    raise ValueError("❌ No valid (positive, finite) data points remain after filtering.")

# Avoid log(0) by epsilon
eps = 1e-12
log_CSac = np.log10(CSac + eps)
log_CSpr = np.log10(CSpr + eps)

# --- Compute Metrics ---
Ei = log_CSpr - log_CSac
E = Ei.mean()
SDE = Ei.std(ddof=0)

Median = np.median(CSpr)
MAD = np.mean(np.abs(CSpr - Median))
Uncertainty_pct = (MAD * 100.0) / (Median if Median != 0 else eps)

# --- Format Results ---
result_df = pd.DataFrame({
    "Model": [MODEL_NAME] * 5,
    "Metric": ["E", "SDE", "Median", "MAD", "Uncertainty (%)"],
    "Value": [round(E, 6), round(SDE, 6), round(Median, 3), round(MAD, 3), round(Uncertainty_pct, 3)]
})

# Print and copy
print("\nFinal Table 7 (single model):\n")
print(result_df.to_string(index=False))
result_df.to_clipboard(index=False)
print("\n✅ Table copied to clipboard — ready to paste into Excel or Word.")
