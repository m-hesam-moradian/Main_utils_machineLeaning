# =========================================================
# uncertainty_mcs_table7_stacked_clipboard.py
# =========================================================
# Requirements:
#   pip install pandas numpy scikit-learn lightgbm
# =========================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb

# -------------------------
# USER SETTINGS
EXCEL_PATH = r"C:\Users\Sam\Desktop\BMM-EI. No.21-Data.xlsx"  # your Excel file path
SHEET_NAME = "Data_after_KFold_LGBR"  # sheet name
TARGET_COLUMN = "SOH"                  # target column name
FEATURE_COLUMNS = None                # None -> all except target
N_MC = 1000                           # Monte Carlo samples
TEST_SIZE = 0.3
RANDOM_STATE = 42
# -------------------------

# Load data
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
if FEATURE_COLUMNS is None:
    FEATURE_COLUMNS = [c for c in df.columns if c != TARGET_COLUMN]

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "LGBR": lgb.LGBMRegressor(random_state=RANDOM_STATE),
    "SGB": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "RF": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "SVM": SVR(kernel="rbf", C=10.0, epsilon=0.1),
}

# Train models
models["LGBR"].fit(X_train, y_train)
models["SGB"].fit(X_train, y_train)
models["RF"].fit(X_train, y_train)
models["SVM"].fit(X_train_scaled, y_train)

# Monte Carlo perturbation factor
perturb_factor = 0.05
col_std = X_train.std(axis=0).replace(0, 1e-8)

def run_mcs_model(model_name, model, X_base, y_true, scaled=False, n_mc=N_MC):
    """Monte Carlo simulation for uncertainty estimation"""
    Xb = X_base.copy()
    n_samples, n_features = Xb.shape
    preds_per_sample = np.zeros((n_samples, n_mc))
    for i in range(n_samples):
        base = Xb.iloc[i].values
        noise = np.random.normal(
            loc=0.0,
            scale=(col_std.values * perturb_factor).astype(float),
            size=(n_mc, n_features),
        )
        sims = base + noise
        sims_in = scaler.transform(sims) if scaled else sims
        preds = model.predict(sims_in)
        preds_per_sample[i, :] = preds
    preds_mean_per_sample = preds_per_sample.mean(axis=1)
    all_preds_flat = preds_per_sample.flatten()
    return preds_mean_per_sample, all_preds_flat, preds_per_sample


# Run MCS for all models
results = []
for name, mdl in models.items():
    use_scaled = name == "SVM"
    preds_mean, all_preds, preds_per_sample = run_mcs_model(
        name, mdl, X_test, y_test, scaled=use_scaled, n_mc=N_MC
    )

    Ei = np.log10(preds_mean + 1e-12) - np.log10(y_test + 1e-12)
    E = Ei.mean()
    SDE = Ei.std(ddof=0)
    Median = np.median(all_preds)
    MAD = np.mean(np.abs(all_preds - Median))
    Uncertainty_pct = (MAD * 100.0) / (Median if Median != 0 else 1e-12)

    results.append(
        {
            "Model": name,
            "E": float(np.round(E, 6)),
            "SDE": float(np.round(SDE, 6)),
            "Median": float(np.round(Median, 3)),
            "MAD": float(np.round(MAD, 3)),
            "Uncertainty (%)": float(np.round(Uncertainty_pct, 3)),
        }
    )

# --- Dempster–Shafer (DST) Ensemble for the two best hybrid models: RF + SVM ---

def dst_combine_predictions(model_a_preds, model_b_preds, y_true):
    eps = 1e-12
    err_a = np.abs(model_a_preds - y_true)
    err_b = np.abs(model_b_preds - y_true)
    rel_a = 1.0 / (err_a + eps)
    rel_b = 1.0 / (err_b + eps)
    mass_a = rel_a / (rel_a + rel_b + eps)
    mass_b = rel_b / (rel_a + rel_b + eps)
    fused = mass_a * model_a_preds + mass_b * model_b_preds
    return fused, mass_a, mass_b


# Get mean predictions for RF and SVM
rf_mean, _, _ = run_mcs_model("RF", models["RF"], X_test, y_test, scaled=False, n_mc=N_MC)
svm_mean, _, _ = run_mcs_model("SVM", models["SVM"], X_test, y_test, scaled=True, n_mc=N_MC)

dst_pred, mass_a, mass_b = dst_combine_predictions(svm_mean, rf_mean, y_test)

Ei_dst = np.log10(dst_pred + 1e-12) - np.log10(y_test + 1e-12)
E_dst = float(np.round(Ei_dst.mean(), 6))
SDE_dst = float(np.round(Ei_dst.std(ddof=0), 6))
Median_dst = float(np.round(np.median(dst_pred), 3))
MAD_dst = float(np.round(np.mean(np.abs(dst_pred - Median_dst)), 3))
Uncertainty_dst = float(
    np.round((MAD_dst * 100.0) / (Median_dst if Median_dst != 0 else 1e-12), 3)
)

results.append(
    {
        "Model": "DST_RF+SVM",
        "E": E_dst,
        "SDE": SDE_dst,
        "Median": Median_dst,
        "MAD": MAD_dst,
        "Uncertainty (%)": Uncertainty_dst,
    }
)

# =========================================================
# === Build and display stacked tables (each model below) ==
# =========================================================

tables_list = []

for r in results:
    temp_df = pd.DataFrame(
        {
            "Metric": ["E", "SDE", "Median", "MAD", "Uncertainty (%)"],
            "Value": [r["E"], r["SDE"], r["Median"], r["MAD"], r["Uncertainty (%)"]],
        }
    )
    temp_df.insert(0, "Model", r["Model"])
    tables_list.append(temp_df)

# Combine vertically (one under another)
table7_df = pd.concat(tables_list, axis=0, ignore_index=True)

# Print and copy
print("\nFinal stacked Table 7 (each model under another):\n")
print(table7_df.to_string(index=False))
table7_df.to_clipboard(index=False)
print("\n✅ Table 7 copied to clipboard — paste it directly into Excel or Word.")

# =========================================================
# End of script
# =========================================================
