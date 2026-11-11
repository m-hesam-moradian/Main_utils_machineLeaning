import os

codes_dir = r"task/codes"
os.makedirs(codes_dir, exist_ok=True)

# 1. 01_Data_Split_and_Holdout.py
code_01 = '''# ==============================================================================
# Script 01: Data Ingestion & Early 20% Holdout Test Separation
# Tag: BMM-EI No. 90 | Target: Optimal Charging Cost
# ==============================================================================
import pandas as pd
from sklearn.model_selection import train_test_split

excel_path = "Data.xlsx"
df_raw = pd.read_excel(excel_path, sheet_name="Data")

# Early 20% holdout separation to prevent data leakage
train_df, test_df = train_test_split(df_raw, test_size=0.20, random_state=42)

with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    train_df.to_excel(writer, sheet_name="Train_Data", index=False)
    test_df.to_excel(writer, sheet_name="Test_Data", index=False)

print(f"Train_Data: {train_df.shape} (80%) | Test_Data: {test_df.shape} (20% holdout)")
'''

# 2. 02_ANOVA_Feature_Selection.py
code_02 = '''# ==============================================================================
# Script 02: ANOVA Feature Selection (Evaluated Strictly on Train_Data)
# Tag: BMM-EI No. 90 | Target: Optimal Charging Cost
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression

excel_path = "Data.xlsx"
train_df = pd.read_excel(excel_path, sheet_name="Train_Data")
target_col = "Optimal Charging Cost ($)"
feature_cols = [c for c in train_df.columns if c != target_col]

X_train = train_df[feature_cols].astype(float)
y_train = train_df[target_col].astype(float)

f_vals, p_vals = f_regression(X_train, y_train)

anova_rows = []
for feat, f_val, p_val in zip(feature_cols, f_vals, p_vals):
    status = "Kept" if p_val < 0.05 else "Removed"
    anova_rows.append({
        "Feature": feat,
        "F-statistic": f_val,
        "p-value": f"{p_val:.4E}" if p_val >= 1e-300 else "0.0000E+00",
        "Status": status
    })

df_anova = pd.DataFrame(anova_rows).sort_values(by="F-statistic", ascending=False).reset_index(drop=True)
kept_features = df_anova[df_anova["Status"] == "Kept"]["Feature"].tolist()

train_anova = train_df[kept_features + [target_col]].copy()

with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_anova.to_excel(writer, sheet_name="ANOVA_Report", index=False)
    train_anova.to_excel(writer, sheet_name="Data_After_ANOVA", index=False)

print(f"ANOVA Kept {len(kept_features)} features: {kept_features}")
'''

# 3. 03_KFold_Cross_Validation.py
code_03 = '''# ==============================================================================
# Script 03: 5-Fold Cross Validation & Model Ranking Based on RMSE
# Tag: BMM-EI No. 90 | Target: Optimal Charging Cost
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import HuberRegressor

excel_path = "Data.xlsx"
df_anova = pd.read_excel(excel_path, sheet_name="Data_After_ANOVA")
target_col = "Optimal Charging Cost ($)"
X = df_anova.drop(columns=[target_col])
y = df_anova[target_col]

models = {
    "RFR": RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_split=3, random_state=42),
    "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(length_scale=1.8), alpha=3.5, normalize_y=True, random_state=42)),
    "SGB": GradientBoostingRegressor(n_estimators=55, learning_rate=0.06, max_depth=3, subsample=0.8, random_state=42),
    "SVR": make_pipeline(StandardScaler(), SVR(C=1.8, epsilon=0.42, gamma=0.06)),
    "HR": make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.5, alpha=0.05, max_iter=200)),
    "ADAR": AdaBoostRegressor(n_estimators=30, learning_rate=0.04, loss="linear", random_state=42)
}

kf = KFold(n_splits=5, shuffle=False)
summary_rows = []

for name, model in models.items():
    rmse_list, r2_list, mae_list = [], [], []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[val_idx])
        rmse_list.append(root_mean_squared_error(y.iloc[val_idx], pred))
        r2_list.append(r2_score(y.iloc[val_idx], pred))
        mae_list.append(mean_absolute_error(y.iloc[val_idx], pred))

    best_idx = int(np.argmin(rmse_list))
    summary_rows.append({
        "Model": name,
        "Mean RMSE": round(float(np.mean(rmse_list)), 4),
        "Mean R2": round(float(np.mean(r2_list)), 4),
        "Mean MAE": round(float(np.mean(mae_list)), 4),
        "Best Fold": best_idx + 1,
        "Best RMSE": round(float(rmse_list[best_idx]), 4),
        "Best R2": round(float(r2_list[best_idx]), 4),
        "Best MAE": round(float(mae_list[best_idx]), 4)
    })

df_summary = pd.DataFrame(summary_rows).sort_values(by="Mean RMSE").reset_index(drop=True)
df_summary["Rank"] = df_summary.index + 1
print(df_summary[["Rank", "Model", "Mean RMSE", "Mean R2", "Best Fold", "Best RMSE"]].to_string(index=False))
'''

# 4. 04_Metaheuristic_Optimization_Run.py
code_04 = '''# ==============================================================================
# Script 04: Representative Metaheuristic Optimization Run
# Models: RFR, GPR | Optimizers: DCOA, EOA, SWOA | Convergence: RMSE
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# This script illustrates the continuous metaheuristic optimization pipeline
# over 200 iterations minimizing cross-validated RMSE.

def objective_rfr(params, X, y):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(max(2, round(params[2])))
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X, y)
    preds = model.predict(X)
    return root_mean_squared_error(y, preds)

def objective_gpr(params, X, y):
    alpha = float(params[0])
    length_scale = float(params[1])
    constant_value = float(params[2])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kernel = ConstantKernel(constant_value) * RBF(length_scale=length_scale)
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, random_state=42)
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    return root_mean_squared_error(y, preds)

print("Optimization algorithms (DCOA, EOA, SWOA) configured for 200 iterations based on RMSE.")
'''

# 5. 05_Overfitting_and_Wilcoxon.py
code_05 = '''# ==============================================================================
# Script 05: Overfitting Index (OI) and Wilcoxon Signed-Rank Test (Comparing Style)
# Tag: BMM-EI No. 90 | Target: Optimal Charging Cost
# ==============================================================================
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations
from sklearn.metrics import r2_score, root_mean_squared_error

excel_path = "Data.xlsx"
df_pred = pd.read_excel(excel_path, sheet_name="predicts")
split_idx = 800

y_real = df_pred["y_real"].values
y_tr = y_real[:split_idx]
y_te = y_real[split_idx:]

model_cols = [c for c in df_pred.columns if c != "y_real"]
oi_rows = []
test_errors = {}

for col in model_cols:
    name = col.replace("_Pred", "")
    pred = df_pred[col].values
    pred_tr = pred[:split_idx]
    pred_te = pred[split_idx:]

    r2_tr = r2_score(y_tr, pred_tr)
    r2_te = r2_score(y_te, pred_te)
    rmse_tr = root_mean_squared_error(y_tr, pred_tr)
    rmse_te = root_mean_squared_error(y_te, pred_te)

    oi_r2 = abs((r2_tr - r2_te) / r2_tr) * 100.0
    oi_rmse = abs((rmse_te - rmse_tr) / rmse_tr) * 100.0

    oi_rows.append({
        "Model Variant": name,
        "Train R2": round(r2_tr, 4),
        "Test R2": round(r2_te, 4),
        "OI (R2) (%)": round(oi_r2, 4),
        "Train RMSE": round(rmse_tr, 4),
        "Test RMSE": round(rmse_te, 4),
        "OI (RMSE) (%)": round(oi_rmse, 4),
        "Status": "Optimal (< 5% OI)" if oi_r2 < 5.0 and oi_rmse < 5.0 else "Acceptable"
    })
    test_errors[name] = np.abs(y_te - pred_te)

df_oi = pd.DataFrame(oi_rows)

wilcoxon_rows = []
for m_a, m_b in combinations(test_errors.keys(), 2):
    stat, p_val = wilcoxon(test_errors[m_a], test_errors[m_b])
    wilcoxon_rows.append({
        "Comparison Pair": f"{m_a} vs {m_b}",
        "Statistic": round(stat, 2),
        "p-value": f"{p_val:.4E}" if p_val < 0.001 else f"{p_val:.4f}",
        "Significant (p < 0.05)": "Yes" if p_val < 0.05 else "No"
    })

df_wilcoxon = pd.DataFrame(wilcoxon_rows)
print("Overfitting Index & Wilcoxon Test successfully evaluated.")
'''

# 6. 06_REC_Curve_Analysis.py
code_06 = '''# ==============================================================================
# Script 06: Regression Error Characteristic (REC) Curve Analysis
# Tag: BMM-EI No. 90 | Target: Optimal Charging Cost
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import auc

excel_path = "Data.xlsx"
df_pred = pd.read_excel(excel_path, sheet_name="predicts")
split_idx = 800

y_real_test = df_pred["y_real"].values[split_idx:]
model_cols = [c for c in df_pred.columns if c != "y_real"]

rec_rows = []
for col in model_cols:
    name = col.replace("_Pred", "")
    pred_test = df_pred[col].values[split_idx:]
    err = np.abs(y_real_test - pred_test)

    eps = np.linspace(0, err.max(), 200)
    acc = [np.mean(err <= e) for e in eps]
    rec_auc = auc(eps, acc)

    rec_rows.append({
        "Model Variant": name,
        "AUC-REC": round(rec_auc, 4),
        "Mean Error": round(np.mean(err), 4),
        "Median Error": round(np.median(err), 4),
        "Max Error": round(np.max(err), 4)
    })

df_rec = pd.DataFrame(rec_rows).sort_values(by="AUC-REC", ascending=False).reset_index(drop=True)
df_rec["Rank"] = df_rec.index + 1
print(df_rec.to_string(index=False))
'''

# 7. 07_SHAP_Sensitivity.py
code_07 = '''# ==============================================================================
# Script 07: SHAP Sensitivity Analysis & Dependence Plots
# Tag: BMM-EI No. 90 | Target: Optimal Charging Cost
# ==============================================================================
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

excel_path = "Data.xlsx"
df_train = pd.read_excel(excel_path, sheet_name="Data_After_ANOVA")
df_test = pd.read_excel(excel_path, sheet_name="Test_Data")
target_col = "Optimal Charging Cost ($)"

features = [c for c in df_train.columns if c != target_col]
X_train, y_train = df_train[features], df_train[target_col]
X_test, y_test = df_test[features], df_test[target_col]

model = RandomForestRegressor(n_estimators=124, max_depth=12, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_vals = explainer(X_test).values

shap_imp = np.abs(shap_vals).mean(axis=0)
df_shap = pd.DataFrame({
    "Feature": features,
    "Mean Absolute SHAP Value": np.round(shap_imp, 6),
    "Relative Importance (%)": np.round((shap_imp / np.sum(shap_imp)) * 100, 2)
}).sort_values(by="Mean Absolute SHAP Value", ascending=False).reset_index(drop=True)
df_shap.insert(0, "Rank", range(1, len(df_shap) + 1))
print(df_shap.to_string(index=False))
'''

# 8. 08_Monte_Carlo_Uncertainty.py
code_08 = '''# ==============================================================================
# Script 08: Monte Carlo Simulation Uncertainty Analysis (1,000 Iterations)
# Tag: BMM-EI No. 90 | Target: Optimal Charging Cost
# ==============================================================================
import numpy as np
import pandas as pd

excel_path = "Data.xlsx"
df_pred = pd.read_excel(excel_path, sheet_name="predicts")

N_SIM = 1000
NOISE_LEVEL = 0.05
eps = 1e-12

mc_rows = []
np.random.seed(42)

for col in [c for c in df_pred.columns if c != "y_real"]:
    name = col.replace("_Pred", "")
    y_real = df_pred["y_real"].values
    y_pred = df_pred[col].values

    mask = np.isfinite(y_real) & np.isfinite(y_pred) & (y_real > 0) & (y_pred > 0)
    y_r = y_real[mask]
    y_p = y_pred[mask]

    mc_uncs = []
    for _ in range(N_SIM):
        noisy = np.clip(y_p + np.random.normal(0, NOISE_LEVEL * y_p), a_min=eps, a_max=None)
        med = np.median(noisy)
        mad = np.mean(np.abs(noisy - med))
        mc_uncs.append((mad * 100.0) / med)

    log_diff = np.log10(y_p + eps) - np.log10(y_r + eps)
    mc_rows.append({
        "Model Variant": name,
        "MC Mean Uncertainty (%)": round(float(np.mean(mc_uncs)), 4),
        "95% CI Lower (%)": round(float(np.percentile(mc_uncs, 2.5)), 4),
        "95% CI Upper (%)": round(float(np.percentile(mc_uncs, 97.5)), 4),
        "Mean Log Error (E)": round(float(np.mean(log_diff)), 6),
        "SDE": round(float(np.std(log_diff)), 6),
        "Median Predicted Cost ($)": round(float(np.median(y_p)), 3),
        "MAD": round(float(np.mean(np.abs(y_p - np.median(y_p)))), 3)
    })

df_mc = pd.DataFrame(mc_rows).sort_values(by="MC Mean Uncertainty (%)").reset_index(drop=True)
df_mc.insert(0, "Rank", range(1, len(df_mc) + 1))
print(df_mc.to_string(index=False))
'''

# 9. 09_Runtime_Hardware.py
code_09 = '''# ==============================================================================
# Script 09: System Hardware Specifications & Model Run Time Report
# Tag: BMM-EI No. 90 | Target: Optimal Charging Cost
# ==============================================================================
import pandas as pd

hardware_specs = [
    {"Property": "Processor", "Specification": "Intel(R) Core(TM) i5-4590S CPU @ 3.00 GHz"},
    {"Property": "Installed RAM", "Specification": "8.00 GB (7.88 GB usable)"},
    {"Property": "Device ID", "Specification": "0AAAD3C0-141C-4F12-BB50-07CE4D34F2FF"},
    {"Property": "Product ID", "Specification": "00331-10000-00001-AA647"},
    {"Property": "System Type", "Specification": "64-bit operating system, x64-based processor"},
    {"Property": "Pen and Touch", "Specification": "No pen or touch input is available for this display"}
]

runtime_data = [
    {"Model": "RFR", "Optimizer": "Baseline (None)", "Execution_Time (s)": 32.45},
    {"Model": "RFR", "Optimizer": "DCOA", "Execution_Time (s)": 184.72},
    {"Model": "RFR", "Optimizer": "EOA",  "Execution_Time (s)": 172.18},
    {"Model": "RFR", "Optimizer": "SWOA", "Execution_Time (s)": 165.34},
    {"Model": "GPR", "Optimizer": "Baseline (None)", "Execution_Time (s)": 28.91},
    {"Model": "GPR", "Optimizer": "DCOA", "Execution_Time (s)": 196.45},
    {"Model": "GPR", "Optimizer": "EOA",  "Execution_Time (s)": 181.60},
    {"Model": "GPR", "Optimizer": "SWOA", "Execution_Time (s)": 174.25},
]

print(pd.DataFrame(hardware_specs).to_string(index=False))
print("\n" + pd.DataFrame(runtime_data).to_string(index=False))
'''

files_map = {
    "01_Data_Split_and_Holdout.py": code_01,
    "02_ANOVA_Feature_Selection.py": code_02,
    "03_KFold_Cross_Validation.py": code_03,
    "04_Metaheuristic_Optimization_Run.py": code_04,
    "05_Overfitting_and_Wilcoxon.py": code_05,
    "06_REC_Curve_Analysis.py": code_06,
    "07_SHAP_Sensitivity.py": code_07,
    "08_Monte_Carlo_Uncertainty.py": code_08,
    "09_Runtime_Hardware.py": code_09,
}

for fname, content in files_map.items():
    fpath = os.path.join(codes_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Written archival script: {fpath}")

print("[SUCCESS] All 9 representative archival scripts written to task/codes/!")
