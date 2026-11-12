import os
import sys
import numpy as np
import pandas as pd
import openpyxl
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    roc_curve, auc, brier_score_loss
)

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

model_sheets = [
    'LGBC(No_SMOTE)', 'LGBC(No_SMOTE) + GOA', 'LGBC(No_SMOTE) + BOA', 'LGBC(No_SMOTE) + LBOA',
    'ETC(No_SMOTE)', 'ETC(No_SMOTE) + GOA', 'ETC(No_SMOTE) + BOA', 'ETC(No_SMOTE) + LBOA',
    'BC(No_SMOTE)', 'BC(No_SMOTE) + GOA', 'BC(No_SMOTE) + BOA', 'BC(No_SMOTE) + LBOA',
    'LGBC(SMOTE)', 'LGBC(SMOTE) + GOA', 'LGBC(SMOTE) + BOA', 'LGBC(SMOTE) + LBOA',
    'ETC(SMOTE)', 'ETC(SMOTE) + GOA', 'ETC(SMOTE) + BOA', 'ETC(SMOTE) + LBOA',
    'BC(SMOTE)', 'BC(SMOTE) + GOA', 'BC(SMOTE) + BOA', 'BC(SMOTE) + LBOA'
]

print(f"[2] Opening workbook: {excel_path}...")
wb = openpyxl.load_workbook(excel_path)

classes = np.array([0, 1])

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

def get_row_metrics(yt, yp, prob):
    acc = float(accuracy_score(yt, yp))
    return [
        acc,
        float(precision_score(yt, yp, average="macro", zero_division=0)),
        float(recall_score(yt, yp, average="macro", zero_division=0)),
        float(f1_score(yt, yp, average="macro", zero_division=0)),
        float(matthews_corrcoef(yt, yp)),
        float(cohen_kappa_score(yt, yp)),
        1.0 - acc,
        calc_markedness(yt, yp),
        calc_brier(yt, prob)
    ]

results_summary = []

print(f"[3] Processing metrics, CM, and ROC in all {len(model_sheets)} model sheets...")

for sheet_name in model_sheets:
    ws = wb[sheet_name]
    
    # 1. Read existing y_real, y_pred, prob_0, prob_1 from columns A-D
    rows = []
    max_r = ws.max_row
    for r in range(3, max_r + 1):
        yr = ws.cell(r, 1).value
        yp = ws.cell(r, 2).value
        p0 = ws.cell(r, 3).value
        p1 = ws.cell(r, 4).value
        if yr is not None and yp is not None:
            rows.append((int(yr), int(yp), float(p0) if p0 is not None else 0.5, float(p1) if p1 is not None else 0.5))
        else:
            if len(rows) > 0 and yr is None:
                # Stop reading if we hit empty data row
                break
                
    n_total = len(rows)
    is_smote = "SMOTE" in sheet_name and "No_SMOTE" not in sheet_name
    split_idx = 5465 if is_smote else 4000
    n_test = n_total - split_idx
    
    y_real = np.array([x[0] for x in rows])
    y_pred = np.array([x[1] for x in rows])
    y_prob = np.array([[x[2], x[3]] for x in rows])
    
    y_tr, p_tr, prob_tr = y_real[:split_idx], y_pred[:split_idx], y_prob[:split_idx]
    y_te, p_te, prob_te = y_real[split_idx:], y_pred[split_idx:], y_prob[split_idx:]
    
    # Metrics calculations
    metrics_all = get_row_metrics(y_real, y_pred, y_prob)
    metrics_tr = get_row_metrics(y_tr, p_tr, prob_tr)
    metrics_te = get_row_metrics(y_te, p_te, prob_te)
    
    # Class-wise metrics evaluated strictly on Test partition
    prec_pc = precision_score(y_te, p_te, average=None, labels=classes, zero_division=0)
    rec_pc = recall_score(y_te, p_te, average=None, labels=classes, zero_division=0)
    f1_pc = f1_score(y_te, p_te, average=None, labels=classes, zero_division=0)
    cm_te = confusion_matrix(y_te, p_te, labels=classes)
    
    class_rows = {}
    for i, cls in enumerate(classes):
        y_bin = (y_te == cls).astype(int)
        p_bin = (p_te == cls).astype(int)
        
        # One-vs-Rest Accuracy properly accounts for True Negatives: (TP + TN) / N
        acc_ovr = float(accuracy_score(y_bin, p_bin))
        err_ovr = 1.0 - acc_ovr
        mcc_ovr = float(matthews_corrcoef(y_bin, p_bin))
        kappa_ovr = float(cohen_kappa_score(y_bin, p_bin))
        brier_cls = float(brier_score_loss(y_bin, prob_te[:, i]))
        
        # Per-class Markedness: PPV_cls + NPV_cls - 1
        tp = cm_te[i, i]
        fp = cm_te[:, i].sum() - tp
        fn = cm_te[i, :].sum() - tp
        tn = cm_te.sum() - (tp + fp + fn)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        markedness_cls = float(ppv + npv - 1)
        
        class_rows[cls] = [
            acc_ovr,
            float(prec_pc[i]),
            float(rec_pc[i]),
            float(f1_pc[i]),
            mcc_ovr,
            kappa_ovr,
            err_ovr,
            markedness_cls,
            brier_cls
        ]
        
    # --- 1. Update Metrics Table in Rows 3 to 7, Cols 10 to 18 ---
    table_data = {
        3: metrics_all,
        4: metrics_tr,
        5: metrics_te,
        6: class_rows[0],
        7: class_rows[1]
    }
    
    for row_idx, row_vals in table_data.items():
        for col_offset, val in enumerate(row_vals):
            col_idx = 10 + col_offset  # Cols 10 to 18
            ws.cell(row=row_idx, column=col_idx).value = val
            
    # --- 2. Update Confusion Matrix in Rows 15 to 16, Cols 7 to 8 ---
    # CM strictly on Test partition
    ws.cell(row=15, column=7).value = int(cm_te[0, 0])
    ws.cell(row=15, column=8).value = int(cm_te[0, 1])
    ws.cell(row=16, column=7).value = int(cm_te[1, 0])
    ws.cell(row=16, column=8).value = int(cm_te[1, 1])
    
    # Mathematical alignment assertion
    test_acc_table = metrics_te[0]
    cm_diag_acc = np.diag(cm_te).sum() / cm_te.sum()
    assert abs(test_acc_table - cm_diag_acc) < 1e-9, f"Mismatch in {sheet_name}: {test_acc_table} vs {cm_diag_acc}"
    
    # --- 3. Update ROC Curve Table in Rows 15 onwards, Cols 10 to 14 ---
    # Clear old ROC values from row 15 to 50
    for r in range(15, 60):
        for c in range(10, 15):
            ws.cell(row=r, column=c).value = None
            
    # Write new ROC curve on Test partition
    cur_r = 15
    for i, cls in enumerate(classes):
        y_bin = (y_te == cls).astype(int)
        score = prob_te[:, i]
        fpr, tpr, thr = roc_curve(y_bin, score)
        roc_auc = auc(fpr, tpr)
        for j in range(len(fpr)):
            ws.cell(row=cur_r, column=10).value = int(cls)
            ws.cell(row=cur_r, column=11).value = float(fpr[j])
            ws.cell(row=cur_r, column=12).value = float(tpr[j])
            ws.cell(row=cur_r, column=13).value = float(thr[j]) if j < len(thr) and not np.isinf(thr[j]) else ""
            ws.cell(row=cur_r, column=14).value = float(roc_auc) if j == len(fpr) - 1 else None
            cur_r += 1
            
    results_summary.append({
        "Sheet": sheet_name,
        "N_total": n_total,
        "N_test": n_test,
        "Train_Acc": round(metrics_tr[0], 4),
        "Test_Acc": round(metrics_te[0], 4),
        "CM_Diag_Acc": round(cm_diag_acc, 4),
        "Test_F1": round(metrics_te[3], 4),
        "CM_Test": f"[[{cm_te[0,0]},{cm_te[0,1]}],[{cm_te[1,0]},{cm_te[1,1]}]]"
    })
    print(f"  [OK] {sheet_name}: Test Acc={metrics_te[0]:.4f}, CM Diag Acc={cm_diag_acc:.4f}, Test F1={metrics_te[3]:.4f}")

print("\n[4] Saving updated workbook to task/Data.xlsx...")
wb.save(excel_path)
wb.close()

print("\n[SUCCESS] Successfully updated all 24 model sheets in-place!")
print(pd.DataFrame(results_summary).to_string())
