import pandas as pd
import numpy as np

# -------------------------
# MONTE CARLO SETTINGS (Report these to the reviewer)
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
SHEET_NAME = "predicts(VIF)"
N_SIMULATIONS = 1000               # Number of Monte Carlo simulations
NOISE_LEVEL = 0.05                 # 5% noise level
CONFIDENCE_INTERVAL = [2.5, 97.5]  # 95% Confidence Interval boundaries
# -------------------------

# Load data
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=0)

columns = df.columns.tolist()
structured_data = []

for i in range(0, len(columns), 2):
    model_name = columns[i].strip()
    y_real = np.array(df.iloc[:, i], dtype=float)
    y_pred = np.array(df.iloc[:, i + 1], dtype=float)
    
    mask_valid = np.isfinite(y_real) & np.isfinite(y_pred) & (y_real > 0) & (y_pred > 0)
    structured_data.append({"name": model_name, "y_real": y_real[mask_valid], "y_pred": y_pred[mask_valid]})

all_results = []
eps = 1e-12 

# Loop through all models
for entry in structured_data:
    name = entry["name"]
    y_real = entry["y_real"]
    y_pred_original = entry["y_pred"]
    
    if len(y_real) == 0:
        continue

    # Arrays to store the Monte Carlo results for this model
    mc_uncertainties = []
    
    # --- MONTE CARLO SIMULATION LOOP ---
    for _ in range(N_SIMULATIONS):
        # Apply Gaussian perturbation (noise) to the predictions
        noise = np.random.normal(0, NOISE_LEVEL * y_pred_original)
        y_pred_noisy = y_pred_original + noise
        
        # Ensure no negative predictions after noise injection
        y_pred_noisy = np.clip(y_pred_noisy, a_min=eps, a_max=None)
        
        # Calculate Uncertainty (%) for this specific simulation
        Median_noisy = np.median(y_pred_noisy)
        MAD_noisy = np.mean(np.abs(y_pred_noisy - Median_noisy))
        Uncertainty_pct = (MAD_noisy * 100.0) / (Median_noisy if Median_noisy != 0 else eps)
        
        mc_uncertainties.append(Uncertainty_pct)
    
    # Calculate Mean and Confidence Intervals from the simulations
    mean_uncertainty = np.mean(mc_uncertainties)
    ci_lower = np.percentile(mc_uncertainties, CONFIDENCE_INTERVAL[0])
    ci_upper = np.percentile(mc_uncertainties, CONFIDENCE_INTERVAL[1])
    
    all_results.append({
        "Model": name, 
        "MC Mean Uncertainty (%)": round(mean_uncertainty, 3),
        "95% CI Lower": round(ci_lower, 3),
        "95% CI Upper": round(ci_upper, 3)
    })

# Convert to DataFrame and copy to clipboard
result_df = pd.DataFrame(all_results)
print("\nMonte Carlo Uncertainty Table:\n")
print(result_df.to_string(index=False))
result_df.to_clipboard(index=False)
print("\n✅ Table copied to clipboard.")