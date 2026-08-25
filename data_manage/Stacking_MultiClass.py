import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, auc,
    cohen_kappa_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
import win32com.client

Models_names = ["MLR + POA", "MLR + HEOA"]


# =====================================================================
# 0. EXCEL UTILITIES
# =====================================================================
def close_excel_file(filepath):
    """Safely closes the target Excel workbook if currently open in Excel."""
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print(f"[OK] Closed open Excel file: {filepath}")
                break
    except Exception:
        pass


# =====================================================================
# 1. METRICS FUNCTION (Adapted for Multi-Class Classification)
# =====================================================================
def build_classification_reports(y_real, y_pred, y_pred_prob, split_ratio=0.8):
    y_real = np.asarray(y_real).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_pred_prob = np.asarray(y_pred_prob)
    
    classes = np.unique(y_real)
    num_classes = len(classes)

    def calculate_markedness(y_true, y_hat):
        cm = confusion_matrix(y_true, y_hat, labels=classes)
        with np.errstate(divide='ignore', invalid='ignore'):
            ppv = np.diag(cm) / cm.sum(axis=0)
            npvs = []
            for i in range(len(classes)):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
                npvs.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
            markedness_per_class = np.nan_to_num(ppv) + np.array(npvs) - 1
            return np.mean(markedness_per_class)

    def calculate_brier_score(y_true, y_prob):
        try:
            brier_list = []
            for i, cls in enumerate(classes):
                y_bin = (y_true == cls).astype(int)
                if i < y_prob.shape[1]:
                    brier_list.append(brier_score_loss(y_bin, y_prob[:, i]))
            return np.mean(brier_list) if brier_list else 0.0
        except Exception:
            return 0.0

    def get_metrics(y_true, y_hat, y_prob):
        acc = accuracy_score(y_true, y_hat)
        return {
            "Accuracy": acc,
            "Precision": precision_score(y_true, y_hat, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_hat, average="macro", zero_division=0),
            "F1": f1_score(y_true, y_hat, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_hat),
            "Kappa": cohen_kappa_score(y_true, y_hat),
            "Class-Wise Error": 1.0 - acc,
            "Markedness": calculate_markedness(y_true, y_hat),
            "Brier Score": calculate_brier_score(y_true, y_prob)
        }

    # --- Train/Test split (80/20 standard)
    split = int(len(y_real) * split_ratio)
    y_real_train, y_real_test = y_real[:split], y_real[split:]
    y_pred_train, y_pred_test = y_pred[:split], y_pred[split:]
    y_prob_train, y_prob_test = y_pred_prob[:split], y_pred_prob[split:]

    cols = ["Set", "Accuracy", "Precision", "Recall", "F1", "MCC", "Kappa", "Class-Wise Error", "Markedness", "Brier Score"]

    df_main = pd.DataFrame([
        ["All", *get_metrics(y_real, y_pred, y_pred_prob).values()],
        ["Train", *get_metrics(y_real_train, y_pred_train, y_prob_train).values()],
        ["Test", *get_metrics(y_real_test, y_pred_test, y_prob_test).values()],
    ], columns=cols)

    # --- Per-class metrics
    precision_pc = precision_score(y_real, y_pred, average=None, labels=classes, zero_division=0)
    recall_pc = recall_score(y_real, y_pred, average=None, labels=classes, zero_division=0)
    f1_pc = f1_score(y_real, y_pred, average=None, labels=classes, zero_division=0)

    cm_all = confusion_matrix(y_real, y_pred, labels=classes)
    markedness_pc = []
    for i in range(len(classes)):
        tp = cm_all[i, i]
        fp = cm_all[:, i].sum() - tp
        fn = cm_all[i, :].sum() - tp
        tn = cm_all.sum() - (tp + fp + fn)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        markedness_pc.append(ppv + npv - 1)

    acc_pc, err_pc, brier_pc, kappa_pc = [], [], [], []
    for i, cls in enumerate(classes):
        idx = y_real == cls
        acc = accuracy_score(y_real[idx], y_pred[idx]) if np.sum(idx) > 0 else 0.0
        acc_pc.append(acc)
        err_pc.append(1.0 - acc)
        
        y_real_bin = (y_real == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        kappa_pc.append(cohen_kappa_score(y_real_bin, y_pred_bin))
        
        if i < y_pred_prob.shape[1]:
            brier_pc.append(brier_score_loss(y_real_bin, y_pred_prob[:, i]))
        else:
            brier_pc.append(np.nan)

    df_class = pd.DataFrame({
        "Set": [f"Class {c}" for c in classes],
        "Accuracy": acc_pc,
        "Precision": precision_pc,
        "Recall": recall_pc,
        "F1": f1_pc,
        "MCC": ["" for _ in classes],
        "Kappa": kappa_pc,
        "Class-Wise Error": err_pc,
        "Markedness": markedness_pc,
        "Brier Score": brier_pc
    })

    df_combined = pd.concat([df_main, df_class], ignore_index=True)

    # --- Confusion Matrix
    cm = confusion_matrix(y_real, y_pred, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Predicted {c}" for c in classes]
    )

    # --- ROC & AUC (One-Vs-Rest)
    roc_rows = []
    for i, cls in enumerate(classes):
        y_true_bin = (y_real == cls).astype(int)
        if i < y_pred_prob.shape[1]:
            y_score = y_pred_prob[:, i]
            fpr, tpr, thr = roc_curve(y_true_bin, y_score)
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
    return df_combined, roc_df, cm_df


# =====================================================================
# 2. DATA LOADING DIRECTLY FROM 'Probs' SHEET IN Data.xlsx
# =====================================================================

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

# Detect whether sheet is named 'Probs' or 'Probs(SMOTE)'
xl = pd.ExcelFile(excel_path)
available_sheets = xl.sheet_names
probs_sheet_name = "Probs" if "Probs" in available_sheets else "Probs(SMOTE)"
print(f"Reading predictions and probabilities from sheet '{probs_sheet_name}' in {excel_path}...")

df_raw = pd.read_excel(excel_path, sheet_name=probs_sheet_name)
# Remove any empty or unnamed padding columns
df_probs = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')].copy()

# Automatically discover available models in the Probs sheet
discovered_models = []
for c in df_probs.columns:
    if c.endswith("_y_real") or c.endswith("_y_pred"):
        prefix = c.rsplit("_y_", 1)[0]
        if prefix not in discovered_models:
            discovered_models.append(prefix)

print(f"Discovered models in sheet '{probs_sheet_name}': {discovered_models}")

# Configure which models to stack (e.g. ["ETC", "MLR"] or choose from discovered_models)
selected_models = Models_names

# Ensure selected models exist in the sheet
for m in selected_models:
    if m not in discovered_models:
        raise ValueError(f"Model '{m}' not found in '{probs_sheet_name}' sheet columns. Available: {discovered_models}")

# Filter relevant columns and clean NaNs
needed_cols = [f"{selected_models[0]}_y_real"]
for m in selected_models:
    needed_cols.append(f"{m}_y_pred")
    prob_cols_m = [c for c in df_probs.columns if c.startswith(f"{m}_prob_")]
    needed_cols.extend(prob_cols_m)

df_clean = df_probs[needed_cols].dropna().reset_index(drop=True)
print(f"Cleaned samples count: {len(df_clean)}")

# Target ground truth
y_real = df_clean[f"{selected_models[0]}_y_real"].values.astype(int)
classes = np.unique(y_real)
num_classes = len(classes)

# Extract predictions and probabilities for each selected base model
base_predictions = {}
base_probabilities = {}

for m in selected_models:
    y_pred_m = df_clean[f"{m}_y_pred"].values.astype(int)
    prob_cols_m = [f"{m}_prob_{i}" for i in range(num_classes)]
    
    # Fallback search if prob columns use another naming pattern
    if not all(col in df_clean.columns for col in prob_cols_m):
        prob_cols_m = [c for c in df_clean.columns if c.startswith(f"{m}_prob_")]
        
    probs_m = df_clean[prob_cols_m].values
    
    base_predictions[m] = y_pred_m
    base_probabilities[m] = probs_m

# --- BASE MODEL SANITY CHECK ---
print("\n" + "="*60)
print("--- BASE MODEL PERFORMANCE SUMMARY (FROM PROBS SHEET) ---")
for m in selected_models:
    acc = accuracy_score(y_real, base_predictions[m])
    print(f"Model [{m:<10}] Overall Accuracy : {acc:.4f}")
print(f"Target Classes: {classes} (Total {num_classes} Classes)")
print("="*60 + "\n")


# =====================================================================
# 3. MULTI-CLASS STACKING META-MODEL & PROBABILITY MERGING
# =====================================================================

# 1. Stack the extracted class probabilities as Meta-Features
# Using the full probability distributions from the base models in Probs sheet
meta_features = [base_probabilities[m] for m in selected_models]
X_meta = np.hstack(meta_features)

# 2. Train/Test Split (80/20 train/test split without shuffle)
split = int(len(y_real) * 0.8)
X_meta_train, X_meta_test = X_meta[:split], X_meta[split:]
y_real_train, y_real_test = y_real[:split], y_real[split:]

# 3. Train the Stacking Meta-Classifier
# LogisticRegression (Multinomial) produces well-calibrated merged probability distributions
meta_clf = LogisticRegression(max_iter=1000, random_state=42)
print("Training Stacking Meta-Classifier on extracted probabilities...")
meta_clf.fit(X_meta_train, y_real_train)

# 4. Predict fused multi-class labels & merged probability distributions
y_pred_stack = meta_clf.predict(X_meta)
y_prob_stack = meta_clf.predict_proba(X_meta)


# =====================================================================
# 4. MERGE BASE & STACKING PREDICTIONS + PROBABILITIES
# =====================================================================

# Build fused DataFrame containing y_real, all base models, and stacking outputs
fused_dict = {"y_real": y_real}

for m in selected_models:
    fused_dict[f"{m}_y_pred"] = base_predictions[m]
    for i, cls in enumerate(classes):
        fused_dict[f"{m}_prob_{cls}"] = base_probabilities[m][:, i]

# Add merged Stacking predictions and probabilities
fused_dict["Stacking_y_pred"] = y_pred_stack
for i, cls in enumerate(classes):
    fused_dict[f"Stacking_prob_{cls}"] = y_prob_stack[:, i]

df_fused = pd.DataFrame(fused_dict)

# Standard pipeline .npt format: [y_real, y_pred_stack, prob_0, prob_1, ...]
stacking_npt_cols = ["y_real", "y_pred"] + [f"Prob_Class_{cls}" for cls in classes]
df_stacking_npt = pd.DataFrame(
    np.column_stack([y_real, y_pred_stack, y_prob_stack]),
    columns=stacking_npt_cols
)


# =====================================================================
# 5. GENERATE COMPREHENSIVE REPORTS & EXPORT
# =====================================================================

print("\nGenerating Multi-Class Classification Reports for Stacking...")
df_metrics, df_roc, df_cm = build_classification_reports(y_real, y_pred_stack, y_prob_stack)

print("\n--- Stacking Metrics Summary ---")
print(df_metrics.to_string(index=False))

print("\n--- Confusion Matrix ---")
print(df_cm)

print("\n--- Sample Fused Predictions & Merged Probabilities (First 5 rows) ---")
print(df_fused.head())

# --- EXPORT ACTIONS ---

# 1. Copy fused predictions and merged probabilities to Clipboard
df_fused.to_clipboard(index=False)
print("\n[OK] Fused predictions & merged probabilities copied to clipboard.")

# 2. Save .npt export for downstream pipeline compatibility
npt_export_path = r"C:\Users\Sam\Desktop\ML\data\stacking_multiclass.npt"
np.savetxt(npt_export_path, df_stacking_npt.values, fmt="%.6f", delimiter="\t")
print(f"[OK] Stacking .npt output saved to: {npt_export_path}")

# 3. Export full multi-sheet report to Excel
output_excel = "Stacking_MultiClass_Report.xlsx"
close_excel_file(output_excel)
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_fused.to_excel(writer, sheet_name="Predictions_and_Probs", index=False)
    df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    df_cm.to_excel(writer, sheet_name="Confusion_Matrix", index=True)
    df_roc.to_excel(writer, sheet_name="ROC_AUC", index=False)
    df_stacking_npt.to_excel(writer, sheet_name="Stacking_NPT_Format", index=False)

print(f"[OK] Full multi-class stacking report saved to: {output_excel}")
