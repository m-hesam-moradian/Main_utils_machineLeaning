import os
import sys
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    roc_curve, auc, brier_score_loss
)
from scipy.stats import wilcoxon, friedmanchisquare
from itertools import combinations

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

def make_style(color):
    return {
        "font": Font(bold=True),
        "alignment": Alignment(horizontal="center", vertical="center"),
        "fill": PatternFill(start_color=color, end_color=color, fill_type="solid")
    }

header_styles = {
    "value_pred": make_style("9DC3E6"),
    "params": make_style("A9D08E"),
    "metrics": make_style("F4B084"),
    "error": make_style("FFD966"),
    "rec_curve": make_style("FFD966"), # Yellow for Confusion Matrix
    "roc": make_style("9DC3E6"),
    "cm": make_style("FFD966")
}

def write_table(df, startrow, startcol, style_key, worksheet):
    style = header_styles.get(style_key, make_style("D9D9D9"))

    for col_num, col_name in enumerate(df.columns):
        row = startrow + 1
        col = startcol + col_num + 1
        cell = worksheet.cell(row=row, column=col)
        cell.value = str(col_name)
        cell.font = style["font"]
        cell.alignment = style["alignment"]
        cell.fill = style["fill"]

    for row_num, row_data in enumerate(df.values):
        for col_num, value in enumerate(row_data):
            cell = worksheet.cell(row=startrow + 2 + row_num, column=startcol + col_num + 1)
            cell.value = value
            if isinstance(value, (int, np.integer)):
                cell.alignment = Alignment(horizontal="center")
            elif isinstance(value, (float, np.floating)):
                cell.alignment = Alignment(horizontal="right")

def compute_clean_classification_reports(y_real, y_pred, y_prob, split_idx, classes):
    n_total = len(y_real)
    y_tr, p_tr, prob_tr = y_real[:split_idx], y_pred[:split_idx], y_prob[:split_idx]
    y_te, p_te, prob_te = y_real[split_idx:], y_pred[split_idx:], y_prob[split_idx:]

    def calc_markedness(yt, yp):
        cm = confusion_matrix(yt, yp, labels=classes)
        with np.errstate(divide='ignore', invalid='ignore'):
            ppv = np.diag(cm) / cm.sum(axis=0)
            npvs = []
            for i in range(len(classes)):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
                npvs.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
            markedness_pc = np.nan_to_num(ppv) + np.array(npvs) - 1
            return float(np.mean(markedness_pc))

    def calc_brier(yt, prob):
        try:
            brier_list = []
            for i, cls in enumerate(classes):
                y_bin = (np.array(yt) == cls).astype(int)
                if i < prob.shape[1]:
                    brier_list.append(brier_score_loss(y_bin, prob[:, i]))
            return float(np.mean(brier_list)) if brier_list else 0.0
        except Exception:
            return 0.0

    def get_metrics_dict(yt, yp, prob):
        acc = float(accuracy_score(yt, yp))
        return {
            "Accuracy": acc,
            "Precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
            "Recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
            "F1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "MCC": float(matthews_corrcoef(yt, yp)),
            "Kappa": float(cohen_kappa_score(yt, yp)),
            "Class-Wise Error": 1.0 - acc,
            "Markedness": calc_markedness(yt, yp),
            "Brier Score": calc_brier(yt, prob)
        }

    cols = ["Set", "Accuracy", "Precision", "Recall", "F1", "MCC", "Kappa", "Class-Wise Error", "Markedness", "Brier Score"]

    df_main = pd.DataFrame([
        ["All", *get_metrics_dict(y_real, y_pred, y_prob).values()],
        ["Train", *get_metrics_dict(y_tr, p_tr, prob_tr).values()],
        ["Test", *get_metrics_dict(y_te, p_te, prob_te).values()],
    ], columns=cols)

    # Class-wise metrics evaluated strictly on Test partition
    prec_pc = precision_score(y_te, p_te, average=None, labels=classes, zero_division=0)
    rec_pc = recall_score(y_te, p_te, average=None, labels=classes, zero_division=0)
    f1_pc = f1_score(y_te, p_te, average=None, labels=classes, zero_division=0)

    cm_te = confusion_matrix(y_te, p_te, labels=classes)

    acc_pc, mcc_pc, kappa_pc, err_pc, brier_pc, markedness_pc = [], [], [], [], [], []
    for i, cls in enumerate(classes):
        y_bin = (y_te == cls).astype(int)
        p_bin = (p_te == cls).astype(int)

        # One-vs-Rest Accuracy properly accounts for True Negatives: (TP + TN) / N
        acc_ovr = float(accuracy_score(y_bin, p_bin))
        acc_pc.append(acc_ovr)
        err_pc.append(1.0 - acc_ovr)

        # One-vs-Rest MCC and Kappa
        mcc_pc.append(float(matthews_corrcoef(y_bin, p_bin)))
        kappa_pc.append(float(cohen_kappa_score(y_bin, p_bin)))

        # Per-class Brier score
        brier_pc.append(float(brier_score_loss(y_bin, prob_te[:, i])))

        # Per-class Markedness: PPV_cls + NPV_cls - 1
        tp = cm_te[i, i]
        fp = cm_te[:, i].sum() - tp
        fn = cm_te[i, :].sum() - tp
        tn = cm_te.sum() - (tp + fp + fn)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        markedness_pc.append(float(ppv + npv - 1))

    df_class = pd.DataFrame({
        "Set": [f"Class {c}" for c in classes],
        "Accuracy": acc_pc,
        "Precision": [float(x) for x in prec_pc],
        "Recall": [float(x) for x in rec_pc],
        "F1": [float(x) for x in f1_pc],
        "MCC": mcc_pc,
        "Kappa": kappa_pc,
        "Class-Wise Error": err_pc,
        "Markedness": markedness_pc,
        "Brier Score": brier_pc
    })

    df_combined = pd.concat([df_main, df_class], ignore_index=True)

    # Confusion Matrix strictly for Test partition
    cm_df = pd.DataFrame(
        cm_te,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Predicted {c}" for c in classes]
    )

    # ROC curve strictly on Test partition
    roc_rows = []
    for i, cls in enumerate(classes):
        y_bin = (y_te == cls).astype(int)
        score = prob_te[:, i]
        fpr, tpr, thr = roc_curve(y_bin, score)
        roc_auc = auc(fpr, tpr)
        for j in range(len(fpr)):
            roc_rows.append({
                "Class": cls,
                "FPR": fpr[j],
                "TPR": tpr[j],
                "Threshold": thr[j] if j < len(thr) else "",
                "AUC": roc_auc if j == len(fpr) - 1 else ""
            })
    roc_df = pd.DataFrame(roc_rows)

    return df_combined, cm_df, roc_df

# Load existing workbook to extract data, parameters, convergence from each sheet
print("\n[2] Reading existing model sheets from task/Data.xlsx...")
wb_src = load_workbook(excel_path, data_only=True)

model_sheets = [
    'LGBC(No_SMOTE)', 'LGBC(No_SMOTE) + GOA', 'LGBC(No_SMOTE) + BOA', 'LGBC(No_SMOTE) + LBOA',
    'ETC(No_SMOTE)', 'ETC(No_SMOTE) + GOA', 'ETC(No_SMOTE) + BOA', 'ETC(No_SMOTE) + LBOA',
    'BC(No_SMOTE)', 'BC(No_SMOTE) + GOA', 'BC(No_SMOTE) + BOA', 'BC(No_SMOTE) + LBOA',
    'LGBC(SMOTE)', 'LGBC(SMOTE) + GOA', 'LGBC(SMOTE) + BOA', 'LGBC(SMOTE) + LBOA',
    'ETC(SMOTE)', 'ETC(SMOTE) + GOA', 'ETC(SMOTE) + BOA', 'ETC(SMOTE) + LBOA',
    'BC(SMOTE)', 'BC(SMOTE) + GOA', 'BC(SMOTE) + BOA', 'BC(SMOTE) + LBOA'
]

sheets_data = {}

for sheet_name in model_sheets:
    ws = wb_src[sheet_name]

    # 1. Read y_real, y_pred, prob_0, prob_1
    rows_data = []
    for r in ws.iter_rows(min_row=3, max_col=4, values_only=True):
        if r[0] is not None:
            rows_data.append((int(r[0]), int(r[1]), float(r[2]), float(r[3])))
    df_preds = pd.DataFrame(rows_data, columns=["y_real", "y_pred", "prob_0", "prob_1"])

    # 2. Read parameters and values
    param_rows = []
    for r in range(3, 10):
        p_name = ws.cell(row=r, column=6).value
        p_val = ws.cell(row=r, column=7).value
        if p_name is not None and str(p_name).strip() != "":
            param_rows.append({"parameters": p_name, "values": p_val})
    df_params = pd.DataFrame(param_rows)

    # 3. Read convergence if optimizer sheet
    is_optimizer = "+" in sheet_name
    df_conv = None
    if is_optimizer:
        conv_vals = []
        for r in range(3, 203):
            c_val = ws.cell(row=r, column=22).value
            if c_val is not None:
                conv_vals.append(float(c_val))
        if conv_vals:
            df_conv = pd.DataFrame({"Convergence": conv_vals})

    sheets_data[sheet_name] = {
        "df_preds": df_preds,
        "df_params": df_params,
        "df_conv": df_conv,
        "is_optimizer": is_optimizer
    }

wb_src.close()
print(f"Extracted data for {len(sheets_data)} model sheets.")

# Process and re-export each model sheet
print("\n[3] Re-generating clean, mathematically aligned model sheets...")
classes = np.array([0, 1])

with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for sheet_name, s_data in sheets_data.items():
        print(f"--> Processing {sheet_name}...")
        df_preds = s_data["df_preds"]
        df_params = s_data["df_params"]
        df_conv = s_data["df_conv"]
        is_optimizer = s_data["is_optimizer"]

        y_real = df_preds["y_real"].values
        y_pred = df_preds["y_pred"].values
        y_prob = df_preds[["prob_0", "prob_1"]].values

        n_total = len(y_real)
        is_smote = "SMOTE" in sheet_name and "No_SMOTE" not in sheet_name
        split_idx = 5465 if is_smote else 4000
        n_test = n_total - split_idx

        # Generate clean classification reports
        df_combined, cm_df, roc_df = compute_clean_classification_reports(
            y_real, y_pred, y_prob, split_idx, classes
        )

        # Verification check: Test accuracy must equal CM diagonal accuracy
        test_acc_table = df_combined.loc[df_combined["Set"] == "Test", "Accuracy"].values[0]
        cm_test_diag_acc = np.diag(cm_df.values).sum() / cm_df.values.sum()
        assert abs(test_acc_table - cm_test_diag_acc) < 1e-9, f"Mismatch in {sheet_name}!"

        # Partitions
        mid_test = split_idx + (n_test // 2)
        df_train_data = pd.DataFrame({
            "Train_Real": y_real[:split_idx],
            "Train_Pred": y_pred[:split_idx]
        })
        df_test_data = pd.DataFrame({
            "Test_Real": y_real[split_idx:mid_test],
            "Test_Pred": y_pred[split_idx:mid_test]
        })
        df_val_data = pd.DataFrame({
            "Val_Real": y_real[mid_test:],
            "Val_Pred": y_pred[mid_test:]
        })

        # Overwrite worksheet
        if sheet_name in writer.book.sheetnames:
            del writer.book[sheet_name]
        worksheet = writer.book.create_sheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        # Title bar
        merge_end_col = 17 if is_optimizer else 16
        worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=merge_end_col)
        cell = worksheet.cell(row=1, column=1)
        cell.value = sheet_name
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.fill = PatternFill(start_color="E1DFFF", end_color="E1DFFF", fill_type="solid")

        # 1. Value / Preds table (Cols A-D)
        write_table(df_preds, startrow=1, startcol=0, style_key="value_pred", worksheet=worksheet)

        # 2. Parameters table (Cols F-G)
        params_col = 5
        write_table(df_params, startrow=1, startcol=params_col, style_key="params", worksheet=worksheet)

        # 3. Metrics Table (Cols I-R)
        metrics_col = 8
        write_table(df_combined, startrow=1, startcol=metrics_col, style_key="metrics", worksheet=worksheet)

        # 4. Confusion Matrix (Row 14, Col F)
        cm_start_row = 13
        cm_out = cm_df.reset_index()
        cm_out.rename(columns={"index": "Actual"}, inplace=True)
        write_table(cm_out, startrow=cm_start_row, startcol=params_col, style_key="rec_curve", worksheet=worksheet)

        # 5. ROC Curve (Row 14, Col J)
        roc_col = metrics_col + 1
        write_table(roc_df, startrow=cm_start_row, startcol=roc_col, style_key="roc", worksheet=worksheet)

        # 6. Convergence (Col V) if optimizer
        if is_optimizer and df_conv is not None:
            conv_col = 21
            write_table(df_conv, startrow=1, startcol=conv_col, style_key="error", worksheet=worksheet)
            p_start_col = 25
        else:
            p_start_col = 21

        # 7. Partition Data Columns
        write_table(df_train_data, startrow=1, startcol=p_start_col, style_key="value_pred", worksheet=worksheet)
        p_start_col += 3
        write_table(df_test_data, startrow=1, startcol=p_start_col, style_key="value_pred", worksheet=worksheet)
        p_start_col += 3
        write_table(df_val_data, startrow=1, startcol=p_start_col, style_key="value_pred", worksheet=worksheet)

print("[SUCCESS] All 24 model sheets re-exported with mathematical alignment!")

# ================= 4. Regenerate Downstream Analysis Sheets =================
print("\n[4] Updating Downstream Analysis Sheets...")

for scen in ["No_SMOTE", "SMOTE"]:
    print(f"\n--> Updating downstream sheets for {scen}...")
    scen_sheets = [s for s in model_sheets if (scen in s and ("No_SMOTE" in s if scen == "No_SMOTE" else "No_SMOTE" not in s))]

    # 1. predicts({scen})
    df_pred_out = pd.DataFrame()
    df_prob_out = pd.DataFrame()

    first_model = scen_sheets[0]
    y_real_ref = sheets_data[first_model]["df_preds"]["y_real"].values
    df_pred_out["y_real"] = y_real_ref
    df_prob_out["y_real"] = y_real_ref

    for m_name in scen_sheets:
        yp = sheets_data[m_name]["df_preds"]["y_pred"].values
        p0 = sheets_data[m_name]["df_preds"]["prob_0"].values
        p1 = sheets_data[m_name]["df_preds"]["prob_1"].values

        df_pred_out[f"{m_name}_Pred"] = yp

        df_prob_out[f"{m_name}_y_pred"] = yp
        df_prob_out[f"{m_name}_prob_0"] = p0
        df_prob_out[f"{m_name}_prob_1"] = p1

    # 2. Brier Decomposition
    brier_decomp_rows = []
    for m_name in scen_sheets:
        yp = sheets_data[m_name]["df_preds"]["y_pred"].values
        p0 = sheets_data[m_name]["df_preds"]["prob_0"].values
        p1 = sheets_data[m_name]["df_preds"]["prob_1"].values

        y_bin_0 = (y_real_ref == 0).astype(int)
        y_bin_1 = (y_real_ref == 1).astype(int)

        brier_0 = brier_score_loss(y_bin_0, p0)
        brier_1 = brier_score_loss(y_bin_1, p1)
        brier_multi = brier_0 + brier_1

        # Uncertainty: base rate variance
        base_rate_0 = np.mean(y_bin_0)
        base_rate_1 = np.mean(y_bin_1)
        unc_0 = base_rate_0 * (1.0 - base_rate_0)
        unc_1 = base_rate_1 * (1.0 - base_rate_1)

        brier_decomp_rows.append({"Model": m_name, "No.": 1, "Metric": "N", "Value": len(y_real_ref)})
        brier_decomp_rows.append({"Model": m_name, "No.": 2, "Metric": "Labels", "Value": "0,1"})
        brier_decomp_rows.append({"Model": m_name, "No.": 3, "Metric": "Brier_multiclass", "Value": round(brier_multi, 6)})
        brier_decomp_rows.append({"Model": m_name, "No.": 4, "Metric": "Brier_decomp_0_brier", "Value": round(brier_0, 6)})
        brier_decomp_rows.append({"Model": m_name, "No.": 5, "Metric": "Brier_decomp_0_uncertainty", "Value": round(unc_0, 6)})
        brier_decomp_rows.append({"Model": m_name, "No.": 6, "Metric": "Brier_decomp_1_brier", "Value": round(brier_1, 6)})
        brier_decomp_rows.append({"Model": m_name, "No.": 7, "Metric": "Brier_decomp_1_uncertainty", "Value": round(unc_1, 6)})

    df_brier_decomp = pd.DataFrame(brier_decomp_rows)

    # 3. Wilcoxon Test (Pairwise on Test errors)
    split_idx = 5465 if scen == "SMOTE" else 4000
    y_test_ref = y_real_ref[split_idx:]
    wilcox_rows = []

    for m_a, m_b in combinations(scen_sheets, 2):
        err_a = np.abs(y_test_ref - sheets_data[m_a]["df_preds"]["y_pred"].values[split_idx:])
        err_b = np.abs(y_test_ref - sheets_data[m_b]["df_preds"]["y_pred"].values[split_idx:])

        diff = err_a - err_b
        if np.all(diff == 0):
            stat, p_val = 0.0, 1.0
        else:
            try:
                stat, p_val = wilcoxon(err_a, err_b)
            except Exception:
                stat, p_val = 0.0, 1.0

        res_str = "Significant" if p_val < 0.05 else "Not Significant"
        wilcox_rows.append({
            "Comparison": f"{m_a} vs {m_b}",
            "Statistic": round(float(stat), 4),
            "P-Value": round(float(p_val), 6) if p_val >= 1e-6 else f"{p_val:.4e}",
            "Result (alpha=0.05)": res_str
        })
    df_wilcox = pd.DataFrame(wilcox_rows)

    # 4. Clean Friedman Test (Omnibus across all models in scenario)
    test_errors = [np.abs(y_test_ref - sheets_data[m]["df_preds"]["y_pred"].values[split_idx:]) for m in scen_sheets]
    stat_fried, p_fried = friedmanchisquare(*test_errors)

    friedman_rows = [{
        "Comparison": f"Omnibus Across All {len(scen_sheets)} Models ({scen})",
        "Statistic": round(float(stat_fried), 5),
        "P-Value": round(float(p_fried), 6) if p_fried >= 1e-6 else f"{p_fried:.4e}",
        "Significance": "Significant (p < 0.05)" if p_fried < 0.05 else "Not Significant"
    }]
    df_friedman = pd.DataFrame(friedman_rows)

    # Save downstream sheets to task/Data.xlsx
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_pred_out.to_excel(writer, sheet_name=f"predicts({scen})", index=False)
        df_prob_out.to_excel(writer, sheet_name=f"Probs({scen})", index=False)
        df_brier_decomp.to_excel(writer, sheet_name=f"Brier_Decomposition({scen})", index=False)
        df_wilcox.to_excel(writer, sheet_name=f"Wilcoxon_Stats({scen})", index=False)
        df_friedman.to_excel(writer, sheet_name=f"Friedman_Stats({scen})", index=False)

print("\n[SUCCESS] All downstream sheets successfully regenerated!")
