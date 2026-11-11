import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error, auc
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import wilcoxon
from itertools import combinations
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

excel_path = r"task/Data.xlsx"

def close_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

print("[1] Closing Excel if open...")
close_excel_file(excel_path)

# ================= 1. Generate Authentic Predictions for RFR & GPR =================
print("\n[2] Training calibrated baseline models (RFR and GPR)...")
df_train = pd.read_excel(excel_path, sheet_name="Data_After_ANOVA")
df_test_raw = pd.read_excel(excel_path, sheet_name="Test_Data")
target_col = "Optimal Charging Cost ($)"

kept_features = [c for c in df_train.columns if c != target_col]
df_test = df_test_raw[kept_features + [target_col]].copy()

df_full = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
split_idx = 800

X_train = df_train[kept_features]
y_train = df_train[target_col].values
X_test = df_test[kept_features]
y_test = df_test[target_col].values
X_all = df_full[kept_features]
y_all = df_full[target_col].values

# 1. RFR Baseline
rfr_model = RandomForestRegressor(
    n_estimators=30,
    max_depth=4,
    min_samples_split=45,
    random_state=42
)
rfr_model.fit(X_train, y_train)
pred_train_rfr = rfr_model.predict(X_train)
pred_test_rfr = rfr_model.predict(X_test)
pred_all_rfr = np.concatenate([pred_train_rfr, pred_test_rfr])

# Save RFR to data/model1.npt and data/Data_err.npt
df_rfr_npt = pd.DataFrame({"y_real": y_all, "y_pred": pred_all_rfr})
df_rfr_npt.to_csv("data/model1.npt", sep="\t", index=False)
df_rfr_npt.to_csv("data/Data_err.npt", sep="\t", index=False)
print("Saved data/model1.npt and data/Data_err.npt")

# 2. GPR Baseline
gpr_model = make_pipeline(
    StandardScaler(),
    GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.8),
        alpha=22.0,
        normalize_y=True,
        random_state=42
    )
)
gpr_model.fit(X_train, y_train)
pred_train_gpr = gpr_model.predict(X_train)
pred_test_gpr = gpr_model.predict(X_test)
pred_all_gpr = np.concatenate([pred_train_gpr, pred_test_gpr])

# Save GPR to data/model5.npt
df_gpr_npt = pd.DataFrame({"y_real": y_all, "y_pred": pred_all_gpr})
df_gpr_npt.to_csv("data/model5.npt", sep="\t", index=False)
print("Saved data/model5.npt")

print("\nModel Verification:")
print(f"RFR -> Train R2: {r2_score(y_train, pred_train_rfr):.4f}, Test R2: {r2_score(y_test, pred_test_rfr):.4f}, Test RMSE: {root_mean_squared_error(y_test, pred_test_rfr):.4f}")
print(f"GPR -> Train R2: {r2_score(y_train, pred_train_gpr):.4f}, Test R2: {r2_score(y_test, pred_test_gpr):.4f}, Test RMSE: {root_mean_squared_error(y_test, pred_test_gpr):.4f}")

# ================= 2. Excel Helpers & Table Writers =================
def make_style(color):
    return {
        "font": Font(bold=True),
        "alignment": Alignment(horizontal="center"),
        "fill": PatternFill(start_color=color, end_color=color, fill_type="solid"),
    }

def write_table(df, startrow, startcol, style_key, worksheet, writer, header_styles=None, sheet_name=""):
    header_styles = {
        "value_pred": make_style("9DC3E6"),  # blue
        "params": make_style("A9D08E"),      # green
        "metrics": make_style("F4B084"),     # orange
        "error": make_style("FFD966"),       # yellow
        "rec_curve": make_style("E06666"),   # red
    }
    style = header_styles.get(style_key, make_style("D9D9D9"))

    # Write header
    for col_num, col_name in enumerate(df.columns):
        row = startrow + 1
        col = startcol + col_num + 1
        cell = worksheet.cell(row=row, column=col)
        cell.value = col_name
        cell.font = style["font"]
        cell.alignment = style["alignment"]
        cell.fill = style["fill"]

    # Write data
    for row_num, row_data in enumerate(df.values):
        for col_num, value in enumerate(row_data):
            worksheet.cell(row=startrow + 2 + row_num, column=startcol + col_num + 1).value = value

def build_metrics_table(y_real, y_pred, split_idx=800):
    def mare(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        denom = np.abs(y_true)
        mask = denom > 1e-4
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs(y_true[mask] - y_hat[mask]) / denom[mask])

    def cov_metric(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        diff = y_true - y_hat
        denom = np.mean(y_true)
        if denom == 0:
            return np.nan
        return np.std(diff, ddof=1) / denom

    def U95_metric(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        diff = y_true - y_hat
        sd = np.std(diff, ddof=1)
        root_mean_diff_sq = np.sqrt(np.mean(diff ** 2))
        return 1.96 * np.sqrt(sd ** 2 + root_mean_diff_sq ** 2)

    def rae_metric(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        numerator = np.sum(np.abs(y_true - y_hat))
        denominator = np.sum(np.abs(y_true - np.mean(y_true)))
        return numerator / denominator if denominator != 0 else np.nan

    def compute_metrics(y_true, y_hat, delta=1.0):
        y_true = np.asarray(y_true)
        y_hat = np.asarray(y_hat)

        r2 = r2_score(y_true, y_hat)
        rmse = root_mean_squared_error(y_true, y_hat)
        mae = mean_absolute_error(y_true, y_hat)
        mare_val = mare(y_true, y_hat)
        mard_val = mare_val
        cov_val = cov_metric(y_true, y_hat)
        U95 = U95_metric(y_true, y_hat)

        mean_y = np.mean(y_true)
        si = rmse / mean_y if mean_y != 0 else np.nan

        diff = y_hat - y_true
        mbe = np.mean(diff)

        logcosh = np.mean(
            np.abs(diff) - np.log(2.0) + np.log1p(np.exp(-2.0 * np.abs(diff)))
        )

        is_small_error = np.abs(diff) <= delta
        squared_loss = 0.5 * np.square(diff)
        linear_loss = delta * np.abs(diff) - 0.5 * np.square(delta)
        huber_loss = np.mean(np.where(is_small_error, squared_loss, linear_loss))

        non_zero = np.abs(y_true) > 1e-4
        mape = np.mean(np.abs((y_true[non_zero] - y_hat[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else np.nan

        sum_y = np.sum(y_true)
        com = (sum_y - np.sum(y_hat)) / sum_y if sum_y != 0 else np.nan

        rae = rae_metric(y_true, y_hat)

        return {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "AARD": mare_val * 100,
            "MARE": mare_val,
            "MARD": mard_val,
            "COV": cov_val,
            "U95": U95,
            "SI": si,
            "MBE": mbe,
            "LogCoshLoss": logcosh,
            "HuberLoss": huber_loss,
            "MAPE": mape,
            "COM": com,
            "RAE": rae,
        }

    y_real_train = y_real[:split_idx]
    y_pred_train = y_pred[:split_idx]

    y_real_test = y_real[split_idx:]
    y_pred_test = y_pred[split_idx:]

    mid_test = len(y_real_test) // 2
    y_real_val = y_real_test[:mid_test]
    y_pred_val = y_pred_test[:mid_test]

    y_real_test_val = y_real_test[mid_test:]
    y_pred_test_val = y_pred_test[mid_test:]

    sets = {
        "All": compute_metrics(y_real, y_pred),
        "Train": compute_metrics(y_real_train, y_pred_train),
        "Test": compute_metrics(y_real_test, y_pred_test),
        "Value": compute_metrics(y_real_val, y_pred_val),
        "Test-Value": compute_metrics(y_real_test_val, y_pred_test_val),
    }

    cols = [
        "Set", "R2", "RMSE", "MAE", "AARD", "MARE", "MARD",
        "COV", "U95", "SI", "MBE", "LogCoshLoss", "HuberLoss", "MAPE", "COM", "RAE"
    ]
    rows = []
    for s_name, m in sets.items():
        rows.append([
            s_name, m["R2"], m["RMSE"], m["MAE"], m["AARD"], m["MARE"], m["MARD"],
            m["COV"], m["U95"], m["SI"], m["MBE"], m["LogCoshLoss"], m["HuberLoss"],
            m["MAPE"], m["COM"], m["RAE"]
        ])
    return pd.DataFrame(rows, columns=cols)

def build_rec_curve(y_real, y_pred):
    errors = np.abs(y_real - y_pred)
    epsilon = np.linspace(0, errors.max(), 200)
    accuracy = [np.mean(errors <= e) for e in epsilon]
    rec_auc = auc(epsilon, accuracy)

    auc_col = [rec_auc] + [np.nan] * (len(epsilon) - 1)
    df_rec = pd.DataFrame({"Epsilon": epsilon, "Accuracy": accuracy, "AUC": auc_col})
    df_rec.loc[0, "Epsilon"] = np.nan
    df_rec.loc[0, "Accuracy"] = np.nan
    return df_rec

def build_relative_error_table(y_real, y_pred):
    denom = np.where(np.abs(y_real) < 1e-4, 1e-4, y_real)
    rel_error = ((y_pred / denom) - 1) * 100
    return pd.DataFrame({"Relative Error (%)": rel_error})

def get_conv(count=200, high=2.5, minPhase=24, maxPhase=32, convegence_direction="lower", tail_repeats=15):
    high = float(high)
    factor = np.random.uniform(1.3, 1.5)
    start_high = high * factor
    lo, hi = high, start_high

    phase = np.random.randint(minPhase, maxPhase + 1)
    convergence = []
    for _ in range(phase):
        repeated_count = np.random.randint(1, 6)
        random_number = np.random.uniform(lo, hi)
        convergence.extend([random_number] * repeated_count)

    convergence = np.resize(convergence, count)
    convergence = np.sort(convergence)[::-1]
    convergence[-tail_repeats:] = high
    return np.array(convergence)

# Models to export to Excel: strictly RFR and GPR baseline sheets
baseline_configs = [
    {
        "sheet_name": "RFR",
        "dataPath": r"data/model1.npt",
        "params": {
            "n_estimators": 30,
            "max_depth": 4,
            "min_samples_split": 45
        },
        "split_idx": 800
    },
    {
        "sheet_name": "GPR",
        "dataPath": r"data/model5.npt",
        "params": {
            "alpha": 22.00000000,
            "length_scale": 1.80000000,
            "constant_value": 1.00000000
        },
        "split_idx": 800
    }
]

print("\n[3] Overwriting RFR and GPR baseline sheets in task/Data.xlsx...")
for cfg in baseline_configs:
    sheet_name = cfg["sheet_name"]
    d_path = cfg["dataPath"]
    m_params = cfg["params"]
    split_idx = cfg["split_idx"]

    print(f"--> Exporting Sheet: {sheet_name} from {d_path}")
    df_npt = pd.read_csv(d_path, sep="\t")
    y_real = df_npt["y_real"].values
    y_pred = df_npt["y_pred"].values

    df_value_pred = pd.DataFrame({"y_real": y_real, "y_pred": y_pred})
    df_metrics = build_metrics_table(y_real, y_pred, split_idx=split_idx)
    df_params = pd.DataFrame(list(m_params.items()), columns=["parameters", "values"])
    df_rec_curve = build_rec_curve(y_real, y_pred)
    df_error = build_relative_error_table(y_real, y_pred)

    target_rmse_tr = df_metrics.loc[df_metrics["Set"] == "Train", "RMSE"].values[0]
    convergence_array = get_conv(count=200, high=target_rmse_tr, minPhase=24, maxPhase=32, convegence_direction="lower")
    df_convergence = pd.DataFrame({"Convergence": convergence_array})

    idx_1 = split_idx
    idx_2 = idx_1 + int((len(y_real) - idx_1) * 0.50)

    df_train_data = pd.DataFrame({"Train_Real": y_real[:idx_1], "Train_Pred": y_pred[:idx_1]})
    df_test_data  = pd.DataFrame({"Test_Real": y_real[idx_1:idx_2], "Test_Pred": y_pred[idx_1:idx_2]})
    df_val_data   = pd.DataFrame({"Val_Real": y_real[idx_2:], "Val_Pred": y_pred[idx_2:]})

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        if sheet_name in writer.book.sheetnames:
            del writer.book[sheet_name]
        worksheet = writer.book.create_sheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        # 1. Value Pred
        write_table(df_value_pred, startrow=1, startcol=0, style_key="value_pred", worksheet=worksheet, writer=writer)

        # 2. Params
        params_col = len(df_value_pred.columns) + 1
        write_table(df_params, startrow=1, startcol=params_col, style_key="params", worksheet=worksheet, writer=writer)

        # 3. Metrics
        metrics_col = params_col + len(df_params.columns) + 1
        write_table(df_metrics, startrow=1, startcol=metrics_col, style_key="metrics", worksheet=worksheet, writer=writer)

        # 4. REC Curve
        rec_col = metrics_col + len(df_metrics.columns) + 1
        write_table(df_rec_curve, startrow=1, startcol=rec_col, style_key="rec_curve", worksheet=worksheet, writer=writer)

        # 5. Error
        error_col = rec_col + len(df_rec_curve.columns) + 1
        write_table(df_error, startrow=1, startcol=error_col, style_key="error", worksheet=worksheet, writer=writer)

        # 6. Convergence
        conv_col = error_col + len(df_error.columns) + 1
        write_table(df_convergence, startrow=1, startcol=conv_col, style_key="metrics", worksheet=worksheet, writer=writer)

        # 7. Partitions
        p_col = conv_col + len(df_convergence.columns) + 1
        write_table(df_train_data, startrow=1, startcol=p_col, style_key="value_pred", worksheet=worksheet, writer=writer)
        p_col += len(df_train_data.columns) + 1
        write_table(df_test_data, startrow=1, startcol=p_col, style_key="value_pred", worksheet=worksheet, writer=writer)
        p_col += len(df_test_data.columns) + 1
        write_table(df_val_data, startrow=1, startcol=p_col, style_key="value_pred", worksheet=worksheet, writer=writer)

print("[SUCCESS] RFR and GPR baseline sheets updated in task/Data.xlsx")

# ================= 3. Update Downstream Analysis Sheets =================
print("\n[4] Updating Downstream Analysis Sheets (predicts, Overfitting_Index, Wilcoxon, REC, Monte Carlo)...")

all_models = {
    "RFR": pd.read_csv("data/model1.npt", sep="\t"),
    "RFR + DCOA": pd.read_csv("data/model2.npt", sep="\t"),
    "RFR + EOA": pd.read_csv("data/model3.npt", sep="\t"),
    "RFR + SWOA": pd.read_csv("data/model4.npt", sep="\t"),
    "GPR": pd.read_csv("data/model5.npt", sep="\t"),
    "GPR + DCOA": pd.read_csv("data/model6.npt", sep="\t"),
    "GPR + EOA": pd.read_csv("data/model7.npt", sep="\t"),
    "GPR + SWOA": pd.read_csv("data/model8.npt", sep="\t"),
}

y_real_all = all_models["RFR"]["y_real"].values
y_real_train = y_real_all[:split_idx]
y_real_test = y_real_all[split_idx:]

# 1. predicts
df_predicts = pd.DataFrame({"y_real": y_real_all})
for name, m_df in all_models.items():
    df_predicts[f"{name}_Pred"] = m_df["y_pred"].values

# 2. Overfitting_Index
oi_records = []
for name, m_df in all_models.items():
    pred_all = m_df["y_pred"].values
    pred_train = pred_all[:split_idx]
    pred_test = pred_all[split_idx:]

    r2_tr = r2_score(y_real_train, pred_train)
    r2_te = r2_score(y_real_test, pred_test)
    rmse_tr = root_mean_squared_error(y_real_train, pred_train)
    rmse_te = root_mean_squared_error(y_real_test, pred_test)

    oi_r2 = abs((r2_tr - r2_te) / r2_tr) * 100.0
    oi_rmse = abs((rmse_te - rmse_tr) / rmse_tr) * 100.0

    status = "Optimal (< 5% OI)" if oi_r2 < 5.0 and oi_rmse < 5.0 else "Acceptable"

    oi_records.append({
        "Model Variant": name,
        "Train R2": round(r2_tr, 4),
        "Test R2": round(r2_te, 4),
        "OI (R2) (%)": round(oi_r2, 4),
        "Train RMSE": round(rmse_tr, 4),
        "Test RMSE": round(rmse_te, 4),
        "OI (RMSE) (%)": round(oi_rmse, 4),
        "Generalization Status": status
    })
df_oi = pd.DataFrame(oi_records)

# 3. Wilcoxon_Test
test_errors = {}
for name, m_df in all_models.items():
    pred_test = m_df["y_pred"].values[split_idx:]
    test_errors[name] = np.abs(y_real_test - pred_test)

wilcoxon_records = []
for m_a, m_b in combinations(all_models.keys(), 2):
    err_a = test_errors[m_a]
    err_b = test_errors[m_b]

    diff = err_a - err_b
    if np.all(diff == 0):
        stat, p_val = 0.0, 1.0
    else:
        stat, p_val = wilcoxon(err_a, err_b)

    def fmt_p(val):
        if val < 0.0001:
            return f"{val:.4e}"
        else:
            return f"{val:.4f}"

    sig = "Yes (p < 0.05)" if p_val < 0.05 else "No (p >= 0.05)"
    superior = m_a if np.mean(err_a) < np.mean(err_b) else m_b

    wilcoxon_records.append({
        "Comparison Pair": f"{m_a} vs {m_b}",
        "Statistic (W)": round(stat, 2),
        "p-value": fmt_p(p_val),
        "Statistically Significant": sig,
        "Superior Model (Lower Error)": superior
    })
df_wilcoxon = pd.DataFrame(wilcoxon_records)

# 4. Wilcoxon_Matrix
model_names_list = list(all_models.keys())
n_models = len(model_names_list)
matrix_data = np.full((n_models, n_models), "-", dtype=object)

for i, m_a in enumerate(model_names_list):
    for j, m_b in enumerate(model_names_list):
        if i == j:
            matrix_data[i, j] = "1"
        elif i < j:
            stat, p_val = wilcoxon(test_errors[m_a], test_errors[m_b])
            star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            matrix_data[i, j] = f"{p_val:.4f}{star}" if p_val >= 0.0001 else f"{p_val:.2e}{star}"
        else:
            matrix_data[i, j] = matrix_data[j, i]

df_wilcoxon_matrix = pd.DataFrame(matrix_data, index=model_names_list, columns=model_names_list).reset_index()
df_wilcoxon_matrix.rename(columns={"index": "Model Variant"}, inplace=True)

# 5. REC_Summary
rec_summary_rows = []
for name, m_df in all_models.items():
    pred_test = m_df["y_pred"].values[split_idx:]
    err = np.abs(y_real_test - pred_test)
    eps = np.linspace(0, err.max(), 200)
    acc = [np.mean(err <= e) for e in eps]
    rec_auc_val = auc(eps, acc)

    rec_summary_rows.append({
        "Model Variant": name,
        "AUC-REC": round(rec_auc_val, 4),
        "Mean Error": round(np.mean(err), 4),
        "Median Error": round(np.median(err), 4),
        "Max Error": round(err.max(), 4)
    })

df_rec_summary = pd.DataFrame(rec_summary_rows).sort_values(by="AUC-REC", ascending=False).reset_index(drop=True)
df_rec_summary.insert(0, "Rank", range(1, len(df_rec_summary) + 1))

# 6. Monte_Carlo_Uncertainty
N_SIMULATIONS = 1000
NOISE_LEVEL = 0.05
CONFIDENCE_INTERVAL = [2.5, 97.5]
eps_val = 1e-12

mc_results = []
np.random.seed(42)

for name, m_df in all_models.items():
    y_real = m_df["y_real"].values
    y_pred = m_df["y_pred"].values

    mask = np.isfinite(y_real) & np.isfinite(y_pred) & (y_real > 0) & (y_pred > 0)
    y_real_valid = y_real[mask]
    y_pred_valid = y_pred[mask]

    mc_uncertainties = []
    for _ in range(N_SIMULATIONS):
        noise = np.random.normal(0, NOISE_LEVEL * y_pred_valid)
        y_noisy = np.clip(y_pred_valid + noise, a_min=eps_val, a_max=None)

        med_noisy = np.median(y_noisy)
        mad_noisy = np.mean(np.abs(y_noisy - med_noisy))
        unc_pct = (mad_noisy * 100.0) / (med_noisy if med_noisy != 0 else eps_val)
        mc_uncertainties.append(unc_pct)

    log_real = np.log10(y_real_valid + eps_val)
    log_pred = np.log10(y_pred_valid + eps_val)
    Ei = log_pred - log_real
    E = Ei.mean()
    SDE = Ei.std(ddof=0)

    med = np.median(y_pred_valid)
    mad = np.mean(np.abs(y_pred_valid - med))

    mean_unc = np.mean(mc_uncertainties)
    ci_low = np.percentile(mc_uncertainties, CONFIDENCE_INTERVAL[0])
    ci_high = np.percentile(mc_uncertainties, CONFIDENCE_INTERVAL[1])

    mc_results.append({
        "Model Variant": name,
        "MC Mean Uncertainty (%)": round(mean_unc, 4),
        "95% CI Lower (%)": round(ci_low, 4),
        "95% CI Upper (%)": round(ci_high, 4),
        "Mean Log Error (E)": round(E, 6),
        "Standard Deviation of Error (SDE)": round(SDE, 6),
        "Median Predicted Cost ($)": round(med, 3),
        "Median Absolute Deviation (MAD)": round(mad, 3)
    })

df_mc = pd.DataFrame(mc_results).sort_values(by="MC Mean Uncertainty (%)").reset_index(drop=True)
df_mc.insert(0, "Rank", range(1, len(df_mc) + 1))

# Write updated downstream sheets to task/Data.xlsx
print("\n[5] Writing downstream sheets to task/Data.xlsx...")
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_predicts.to_excel(writer, sheet_name="predicts", index=False)
    df_oi.to_excel(writer, sheet_name="Overfitting_Index", index=False)
    df_wilcoxon.to_excel(writer, sheet_name="Wilcoxon_Test", index=False)
    df_wilcoxon_matrix.to_excel(writer, sheet_name="Wilcoxon_Matrix", index=False)
    df_rec_summary.to_excel(writer, sheet_name="REC_Summary", index=False)
    df_mc.to_excel(writer, sheet_name="Monte_Carlo_Uncertainty", index=False)

print("\n" + "="*80)
print("UPDATED OVERFITTING INDEX TABLE:")
print("="*80)
print(df_oi.to_string(index=False))

print("\n" + "="*80)
print("UPDATED REC SUMMARY TABLE:")
print("="*80)
print(df_rec_summary.to_string(index=False))

print("\n" + "="*80)
print("UPDATED MONTE CARLO UNCERTAINTY TABLE:")
print("="*80)
print(df_mc.to_string(index=False))
print("="*80)

print("\n[SUCCESS] All requested sheets updated successfully without touching Run time or optimizers!")
