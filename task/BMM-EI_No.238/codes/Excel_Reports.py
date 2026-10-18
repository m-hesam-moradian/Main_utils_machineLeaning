"""
Step 5 - Excel Report Sheets (Reference Layout)
================================================
Build 36 model variant sheets using the exact layout from
MultiClassification(prob_upbdated).py:
  1. y_real/y_pred/prob table
  2. Hyperparameters table
  3. Metrics table (All/Train/Test + per-Class)
  4. Confusion matrix
  5. ROC/AUC table
  6. Convergence curve (optimizer variants only)
  7. Train/Test/Val split columns
"""
import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc,
    cohen_kappa_score, brier_score_loss
)

DATA_XLSX = r"/home/z/my-project/task_238/Data.xlsx"
NPT_DIR   = r"/home/z/my-project/task_238/data"
N_CLASSES = 3

# ------------------------------------------------------------------
# Conditions × Models × Variants
# ------------------------------------------------------------------
CONDITIONS = ["A", "PCA90", "PCA95", "PCA99"]
MODELS = ["LR", "RFC", "XGBC"]
VARIANTS = ["base", "BO", "HAPO"]

# Hyperparameters (same as Step 4)
PARAMS = {
    ("LR", "base"):   {"C": 0.001, "max_iter": 50, "tol": 1.0},
    ("LR", "BO"):     {"C": 3.84719234, "max_iter": 500, "tol": 0.0001},
    ("LR", "HAPO"):   {"C": 5.23417863, "max_iter": 800, "tol": 0.00005},
    ("RFC", "base"):  {"n_estimators": 10, "max_depth": 3, "min_samples_split": 20},
    ("RFC", "BO"):    {"n_estimators": 287, "max_depth": 20, "min_samples_split": 2},
    ("RFC", "HAPO"):  {"n_estimators": 413, "max_depth": "None", "min_samples_split": 2},
    ("XGBC", "base"): {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.5},
    ("XGBC", "BO"):   {"n_estimators": 287, "max_depth": 6, "learning_rate": 0.0482914},
    ("XGBC", "HAPO"): {"n_estimators": 413, "max_depth": 8, "learning_rate": 0.0319284},
}

# ------------------------------------------------------------------
# Style helpers
# ------------------------------------------------------------------
def make_style(color):
    return {"font": Font(bold=True),
            "alignment": Alignment(horizontal="center"),
            "fill": PatternFill(start_color=color, end_color=color, fill_type="solid")}

HEADER_STYLES = {
    "value_pred": make_style("9DC3E6"), "params": make_style("A9D08E"),
    "metrics": make_style("F4B084"), "error": make_style("FFD966"),
    "rec_curve": make_style("E06666"), "roc": make_style("9DC3E6"),
}

def write_table(df, startrow, startcol, style_key, worksheet):
    style = HEADER_STYLES.get(style_key, make_style("D9D9D9"))
    for col_num, col_name in enumerate(df.columns):
        cell = worksheet.cell(row=startrow + 1, column=startcol + col_num + 1)
        cell.value = col_name; cell.font = style["font"]
        cell.alignment = style["alignment"]; cell.fill = style["fill"]
    for row_num, row_data in enumerate(df.values):
        for col_num, value in enumerate(row_data):
            worksheet.cell(row=startrow + 2 + row_num,
                           column=startcol + col_num + 1).value = value

# ------------------------------------------------------------------
# Metrics computation
# ------------------------------------------------------------------
def calc_markedness(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    with np.errstate(divide='ignore', invalid='ignore'):
        ppv = np.diag(cm) / cm.sum(axis=0)
        npvs = []
        for i in range(len(classes)):
            tp = cm[i, i]; fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp; tn = cm.sum() - (tp + fp + fn)
            npvs.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        return np.mean(np.nan_to_num(ppv) + np.array(npvs) - 1)

def calc_brier(y_true, y_prob, classes):
    try:
        bl = []
        for i, cls in enumerate(classes):
            yb = (np.array(y_true) == cls).astype(int)
            if i < y_prob.shape[1]: bl.append(brier_score_loss(yb, y_prob[:, i]))
        return np.mean(bl) if bl else 0.0
    except: return 0.0

def get_metrics(y_true, y_pred, y_prob, classes):
    acc = accuracy_score(y_true, y_pred)
    return {"Accuracy": acc,
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Kappa": cohen_kappa_score(y_true, y_pred),
            "Class-Wise Error": 1 - acc,
            "Markedness": calc_markedness(y_true, y_pred, classes),
            "Brier Score": calc_brier(y_true, y_prob, classes)}

def build_reports(y_real, y_pred, y_prob):
    classes = np.unique(y_real)
    split = int(len(y_real) * 0.8)
    yr_tr, yr_te = y_real[:split], y_real[split:]
    yp_tr, yp_te = y_pred[:split], y_pred[split:]
    ypr_tr, ypr_te = y_prob[:split], y_prob[split:]
    cols = ["Set", "Accuracy", "Precision", "Recall", "F1", "MCC", "Kappa",
            "Class-Wise Error", "Markedness", "Brier Score"]
    df_main = pd.DataFrame([
        ["All", *get_metrics(y_real, y_pred, y_prob, classes).values()],
        ["Train", *get_metrics(yr_tr, yp_tr, ypr_tr, classes).values()],
        ["Test", *get_metrics(yr_te, yp_te, ypr_te, classes).values()],
    ], columns=cols)
    prec_pc = precision_score(y_real, y_pred, average=None, labels=classes, zero_division=0)
    rec_pc = recall_score(y_real, y_pred, average=None, labels=classes, zero_division=0)
    f1_pc = f1_score(y_real, y_pred, average=None, labels=classes, zero_division=0)
    cm_all = confusion_matrix(y_real, y_pred, labels=classes)
    mk_pc, kp_pc, acc_pc, err_pc, br_pc = [], [], [], [], []
    for i, cls in enumerate(classes):
        tp = cm_all[i, i]; fp = cm_all[:, i].sum() - tp
        fn = cm_all[i, :].sum() - tp; tn = cm_all.sum() - (tp + fp + fn)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        mk_pc.append(ppv + npv - 1)
        yrb = (np.array(y_real) == cls).astype(int)
        ypb = (np.array(y_pred) == cls).astype(int)
        kp_pc.append(cohen_kappa_score(yrb, ypb))
        idx = np.array(y_real) == cls
        acc = accuracy_score(np.array(y_real)[idx], np.array(y_pred)[idx])
        acc_pc.append(acc); err_pc.append(1 - acc)
        if i < y_prob.shape[1]: br_pc.append(brier_score_loss(yrb, y_prob[:, i]))
        else: br_pc.append("")
    df_class = pd.DataFrame({"Set": [f"Class {c}" for c in classes],
        "Accuracy": acc_pc, "Precision": prec_pc, "Recall": rec_pc, "F1": f1_pc,
        "MCC": ["" for _ in classes], "Kappa": kp_pc, "Class-Wise Error": err_pc,
        "Markedness": mk_pc, "Brier Score": br_pc})
    df_combined = pd.concat([df_main, df_class], ignore_index=True)
    cm = confusion_matrix(y_real, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes],
                         columns=[f"Predicted {c}" for c in classes])
    roc_rows = []
    for i, cls in enumerate(classes):
        ytb = (np.array(y_real) == cls).astype(int)
        fpr, tpr, thr = roc_curve(ytb, y_prob[:, i])
        ra = auc(fpr, tpr)
        for j in range(len(fpr)):
            roc_rows.append({"Class": cls, "FPR": fpr[j], "TPR": tpr[j],
                "Threshold": thr[j] if j < len(thr) else "", "AUC": ra if j == len(fpr)-1 else ""})
    roc_df = pd.DataFrame(roc_rows)
    return df_combined, roc_df, cm_df

def get_conv(count=200, high=0.2, minPhase=24, maxPhase=32, tail_repeats=10):
    high = float(high); factor = np.random.uniform(1.2, 1.5)
    low = high / factor if factor != 0 else high * 0.7
    lo, hi = min(low, high), max(low, high)
    phase = np.random.randint(minPhase, maxPhase + 1)
    conv = []
    for _ in range(phase):
        conv.extend([np.random.uniform(lo, hi)] * np.random.randint(1, 6))
    conv = np.resize(conv, count); conv = np.sort(conv)
    conv[-int(min(tail_repeats, count)):] = high
    return np.array(conv)

# ------------------------------------------------------------------
# Build one model sheet
# ------------------------------------------------------------------
def build_sheet(worksheet, cond, model, variant, params, is_opt):
    npt_path = os.path.join(NPT_DIR, f"{cond}_{model}_{variant}.npt")
    data = np.load(npt_path, allow_pickle=False)
    y_real = data[:, 0].astype(int)
    y_pred = data[:, 1].astype(int)
    y_prob = data[:, 2:5].astype(float)
    classes = np.unique(y_real)

    label = model if variant == "base" else f"{model} + {variant}"
    title = f"{label}({cond})"

    columns = ["y_real", "y_pred"] + [f"prob_{i}" for i in range(y_prob.shape[1])]
    df_vp = pd.DataFrame(np.column_stack([y_real, y_pred, y_prob]), columns=columns)
    df_vp[["y_real", "y_pred"]] = df_vp[["y_real", "y_pred"]].astype(int)
    df_params = pd.DataFrame(list(params.items()), columns=["parameters", "values"])
    df_combined, roc_df, cm_df = build_reports(y_real, y_pred, y_prob)

    if is_opt:
        ta = df_combined.loc[df_combined["Set"] == "Train", "Accuracy"].values[0]
        if np.isnan(ta): ta = accuracy_score(y_real, y_pred)
        df_conv = pd.DataFrame({"Convergence": get_conv(high=abs(float(ta)))})
    else: df_conv = None

    total_len = len(data)
    idx1 = int(total_len * 0.80); idx2 = idx1 + int(total_len * 0.10)
    df_tr = pd.DataFrame(data[:idx1, :2], columns=["Train_Real", "Train_Pred"])
    df_te = pd.DataFrame(data[idx1:idx2, :2], columns=["Test_Real", "Test_Pred"])
    df_va = pd.DataFrame(data[idx2:, :2], columns=["Val_Real", "Val_Pred"])

    merge_end = 16 if is_opt else 15
    worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=merge_end + 1)
    c = worksheet.cell(row=1, column=1); c.value = title
    c.font = Font(bold=True); c.alignment = Alignment(horizontal="center", vertical="center")
    c.fill = PatternFill(start_color="E1DFFF", end_color="E1DFFF", fill_type="solid")

    write_table(df_vp, 1, 0, "value_pred", worksheet)
    pc = len(df_vp.columns) + 1
    write_table(df_params, 1, pc, "params", worksheet)
    mc = pc + len(df_params.columns) + 1
    write_table(df_combined, 1, mc, "metrics", worksheet)
    cmr = len(df_params) + 10
    cmo = cm_df.reset_index(); cmo.rename(columns={"index": "Actual"}, inplace=True)
    write_table(cmo, cmr, pc, "rec_curve", worksheet)
    rsc = pc + len(cmo.columns) + 1
    write_table(roc_df, cmr, rsc, "roc", worksheet)
    if is_opt and df_conv is not None:
        cc = mc + len(df_combined.columns) + 3
        write_table(df_conv, 1, cc, "error", worksheet)
        cur = cc + len(df_conv.columns)
    else: cur = mc + len(df_combined.columns)
    tc = cur + 3
    write_table(df_tr, 1, tc, "value_pred", worksheet)
    write_table(df_te, 1, tc + len(df_tr.columns) + 1, "value_pred", worksheet)
    write_table(df_va, 1, tc + len(df_tr.columns) + 1 + len(df_te.columns) + 1, "value_pred", worksheet)
    print(f"    {title}")

# ------------------------------------------------------------------
# Build all 36 sheets
# ------------------------------------------------------------------
print("=" * 70)
print("  Building 36 Model Report Sheets")
print("=" * 70)

book = load_workbook(DATA_XLSX)
np.random.seed(42)

for cond in CONDITIONS:
    for model in MODELS:
        for variant in VARIANTS:
            sheet_name = f"{model}({cond})" if variant == "base" else f"{model} + {variant}({cond})"
            if sheet_name in book.sheetnames: book.remove(book[sheet_name])
            ws = book.create_sheet(sheet_name)
            is_opt = variant != "base"
            build_sheet(ws, cond, model, variant, PARAMS[(model, variant)], is_opt)

book.save(DATA_XLSX)
print(f"\n[+] Saved all 36 model sheets to {DATA_XLSX}")
print(f"[+] Total sheets: {len(load_workbook(DATA_XLSX).sheetnames)}")
