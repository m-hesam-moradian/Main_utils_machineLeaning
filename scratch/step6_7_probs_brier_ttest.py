import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from itertools import combinations
from sklearn.metrics import brier_score_loss

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
    except Exception as e:
        pass

def compute_brier_decomposition_multiclass(y_true, prob_matrix, labels=None, n_bins=10):
    if labels is None:
        labels = np.unique(y_true)
    K = len(labels)
    N = len(y_true)
    results = {}
    
    for k, lab in enumerate(labels):
        o_k = (np.array(y_true) == lab).astype(float)
        p_k = np.clip(prob_matrix[:, k], 0.0, 1.0)
        
        # Base Brier score for class k
        bs_k = np.mean((p_k - o_k) ** 2)
        
        # Base rate
        base_rate = np.mean(o_k)
        uncertainty = base_rate * (1.0 - base_rate)
        
        # Binned decomposition
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(p_k, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        
        reliability = 0.0
        resolution = 0.0
        
        for b in range(n_bins):
            mask = (bin_ids == b)
            n_b = np.sum(mask)
            if n_b > 0:
                p_bar_b = np.mean(p_k[mask])
                o_bar_b = np.mean(o_k[mask])
                reliability += (n_b / N) * ((p_bar_b - o_bar_b) ** 2)
                resolution += (n_b / N) * ((o_bar_b - base_rate) ** 2)
                
        results[str(lab)] = {
            "brier": float(bs_k),
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty)
        }
    return results

print("[1] Closing Excel...")
close_excel_file(excel_path)

xl = pd.ExcelFile(excel_path)

scenarios = [
    {
        "tag": "SMOTE",
        "models": ["MLR(SMOTE)", "MLR(SMOTE) + HEOA", "LGBC(SMOTE)", "LGBC(SMOTE) + HEOA"]
    },
    {
        "tag": "No_Bal",
        "models": ["MLR(No_Bal)", "MLR(No_Bal) + HEOA", "LGBC(No_Bal)", "LGBC(No_Bal) + HEOA"]
    }
]

probs_sheets = {}
brier_sheets = {}
predicts_sheets = {}
ttest_sheets = {}

for sc in scenarios:
    tag = sc["tag"]
    model_list = sc["models"]
    print(f"\n================ Processing Scenario: {tag} ================")
    
    # 1. Catch Probabilities
    merged_probs = []
    merged_preds = []
    
    for m_name in model_list:
        df_m = pd.read_excel(xl, sheet_name=m_name, header=1)
        # First 5 columns are y_real, y_pred, prob_0, prob_1, prob_2
        val_pred = df_m.iloc[:, :5].dropna(how="all").copy()
        
        # Prepare prob columns: {m_name}_y_real, {m_name}_y_pred, {m_name}_prob_0, etc.
        val_pred.columns = [f"{m_name}_y_real", f"{m_name}_y_pred", f"{m_name}_prob_0", f"{m_name}_prob_1", f"{m_name}_prob_2"]
        merged_probs.append(val_pred)
        
        # Prepare predict columns: {m_name} (y_real), {m_name} (y_pred)
        df_pred_sub = df_m.iloc[:, :2].dropna(how="all").copy()
        df_pred_sub.columns = [f"{m_name}", f"{m_name}"]
        merged_preds.append(df_pred_sub)
        
    df_probs = pd.concat(merged_probs, axis=1)
    df_preds = pd.concat(merged_preds, axis=1)
    
    probs_sheets[f"Probs({tag})"] = df_probs
    predicts_sheets[f"predicts({tag})"] = df_preds
    
    # 2. Compute Brier Decomposition
    all_brier_reports = []
    for m_name in model_list:
        y_true = df_probs[f"{m_name}_y_real"].values.astype(int)
        prob_cols = [f"{m_name}_prob_0", f"{m_name}_prob_1", f"{m_name}_prob_2"]
        prob_mat = df_probs[prob_cols].values.astype(float)
        
        # Total multiclass Brier score
        labels = np.unique(y_true)
        brier_multi = np.mean([brier_score_loss((y_true == l).astype(int), prob_mat[:, i]) for i, l in enumerate(labels)])
        
        metrics = []
        metrics.append(("N", int(len(y_true))))
        metrics.append(("Labels", ",".join([str(x) for x in labels])))
        metrics.append(("Brier_multiclass", round(float(brier_multi), 8)))
        
        decomp = compute_brier_decomposition_multiclass(y_true, prob_mat, labels=labels, n_bins=10)
        for lab, comp in decomp.items():
            metrics.append((f"Brier_decomp_{lab}_brier", round(comp["brier"], 8)))
            metrics.append((f"Brier_decomp_{lab}_reliability", round(comp["reliability"], 8)))
            metrics.append((f"Brier_decomp_{lab}_resolution", round(comp["resolution"], 8)))
            metrics.append((f"Brier_decomp_{lab}_uncertainty", round(comp["uncertainty"], 8)))
            
        df_rep = pd.DataFrame(metrics, columns=["Metric", "Value"])
        df_rep.insert(0, "No.", range(1, len(df_rep) + 1))
        df_rep.insert(0, "Model", m_name)
        all_brier_reports.append(df_rep)
        
    df_brier_final = pd.concat(all_brier_reports, ignore_index=True)
    brier_sheets[f"Brier_Decomposition({tag})"] = df_brier_final
    
    # 3. Compute Paired t-Test
    predictions = {}
    for i in range(0, df_preds.shape[1], 2):
        col_name = df_preds.columns[i]
        predictions[col_name] = df_preds.iloc[:, i + 1].values.astype(float)
        
    results_stats = {}
    results_p = {}
    alpha = 0.05
    
    for m_a, m_b in combinations(predictions.keys(), 2):
        p_a = predictions[m_a]
        p_b = predictions[m_b]
        t_stat, p_val = ttest_rel(p_a, p_b)
        results_stats[f"{m_a} vs {m_b}"] = t_stat
        results_p[f"{m_a} vs {m_b}"] = p_val
        
    df_stats = pd.DataFrame(results_stats.items(), columns=["Comparison", "t-statistic"])
    df_p = pd.DataFrame(results_p.items(), columns=["Comparison", "P-Value"])
    df_ttest = pd.merge(df_stats, df_p, on="Comparison")
    
    def check_sig(p):
        return "Significant" if float(p) < alpha else "Not Significant"
        
    def fmt_p(p):
        v = float(p)
        return f"{v:.6e}" if v < 0.001 else f"{v:.6f}"
        
    df_ttest["Result (alpha=0.05)"] = df_ttest["P-Value"].apply(check_sig)
    df_ttest["P-Value"] = df_ttest["P-Value"].apply(fmt_p)
    df_ttest["t-statistic"] = df_ttest["t-statistic"].apply(lambda x: round(float(x), 6))
    
    ttest_sheets[f"Statistical_t-test({tag})"] = df_ttest
    print(f"[+] Prepared Probs, Brier Decomposition, predicts, and t-test for {tag}")

print("\n[2] Writing Step 6 & 7 sheets to task/Data.xlsx...")
close_excel_file(excel_path)
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for name, df in probs_sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)
        print(f"  Saved sheet: {name}")
    for name, df in brier_sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)
        print(f"  Saved sheet: {name}")
    for name, df in predicts_sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)
        print(f"  Saved sheet: {name}")
    for name, df in ttest_sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)
        print(f"  Saved sheet: {name}")

print("\n[+] Step 6 and Step 7 sheets saved successfully to task/Data.xlsx!")
