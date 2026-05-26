import pandas as pd
import numpy as np

# -------------------------
# USER SETTINGS
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
SHEET_NAME = "predicts(RFE)"  # same sheet as before
# -------------------------

# Load data
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=0)

# Dynamically extract model predictions
columns = df.columns.tolist()
structured_data = []

for i in range(0, len(columns), 2):
    model_name = columns[i].strip()
    y_real = np.array(df.iloc[:, i], dtype=float)
    y_pred = np.array(df.iloc[:, i + 1], dtype=float)
    
    # Filter valid data points (finite, positive)
    mask_valid = np.isfinite(y_real) & np.isfinite(y_pred) & (y_real > 0) & (y_pred > 0)
    y_real = y_real[mask_valid]
    y_pred = y_pred[mask_valid]
    
    structured_data.append({"name": model_name, "y_real": y_real, "y_pred": y_pred})

# Prepare results container
all_results = []

eps = 1e-12  # small epsilon to avoid log(0)

# Loop through all models
for entry in structured_data:
    name = entry["name"]
    y_real = entry["y_real"]
    y_pred = entry["y_pred"]
    
    if len(y_real) == 0:
        print(f"⚠️ No valid data for {name}, skipping...")
        continue
    
    # Log-transform
    log_real = np.log10(y_real + eps)
    log_pred = np.log10(y_pred + eps)
    
    # Compute uncertainty metrics
    Ei = log_pred - log_real
    E = Ei.mean()
    SDE = Ei.std(ddof=0)
    
    Median = np.median(y_pred)
    MAD = np.mean(np.abs(y_pred - Median))
    Uncertainty_pct = (MAD * 100.0) / (Median if Median != 0 else eps)
    
    # Append results
    all_results.extend([
        {"Model": name, "Metric": "E", "Value": round(E, 6)},
        {"Model": name, "Metric": "SDE", "Value": round(SDE, 6)},
        {"Model": name, "Metric": "Median", "Value": round(Median, 3)},
        {"Model": name, "Metric": "MAD", "Value": round(MAD, 3)},
        {"Model": name, "Metric": "Uncertainty (%)", "Value": round(Uncertainty_pct, 3)},
    ])

# Convert to DataFrame
result_df = pd.DataFrame(all_results)

# Print and copy
print("\nFinal Uncertainty Table:\n")
print(result_df.to_string(index=False))
result_df.to_clipboard(index=False)
print("\n✅ Table copied to clipboard — ready to paste into Excel or Word.")
