import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# =====================================================================
# 1. METRICS FUNCTION (Adapted for multi-class)
# =====================================================================
def build_classification_reports(y_real, y_pred, y_pred_prob):
    classes = np.unique(y_real)

    def calculate_markedness(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
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

    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        return {
            "Accuracy": acc,
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Class-Wise Error": 1 - acc,
            "Markedness": calculate_markedness(y_true, y_pred)
        }

    # --- Train/Test split
    split = int(len(y_real) * 0.8)
    y_real_train, y_real_test = y_real[:split], y_real[split:]
    y_pred_train, y_pred_test = y_pred[:split], y_pred[split:]

    cols = ["Set", "Accuracy", "Precision", "Recall", "F1", "MCC", "Class-Wise Error", "Markedness"]

    df_main = pd.DataFrame([
        ["All", *get_metrics(y_real, y_pred).values()],
        ["Train", *get_metrics(y_real_train, y_pred_train).values()],
        ["Test", *get_metrics(y_real_test, y_pred_test).values()],
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

    acc_pc, err_pc = [], []
    for cls in classes:
        idx = y_real == cls
        if sum(idx) > 0:
            acc = accuracy_score(y_real[idx], y_pred[idx])
        else:
            acc = 0
        acc_pc.append(acc)
        err_pc.append(1 - acc)

    df_class = pd.DataFrame({
        "Set": [f"Class {c}" for c in classes],
        "Accuracy": acc_pc,
        "Precision": precision_pc,
        "Recall": recall_pc,
        "F1": f1_pc,
        "MCC": ["" for _ in classes],
        "Class-Wise Error": err_pc,
        "Markedness": markedness_pc
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


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# =====================================================================
# 2. DATA LOADING & MULTI-CLASS STACKING FUSION (FIXED)
# =====================================================================

# Load data
data = np.loadtxt(r"C:\Users\Sam\Desktop\ML\data\Data_err.npt")
y_real = data[:, 0]
y_pred_lgbr = data[:, 1]
y_pred_sgb = data[:, 2]

classes = np.unique(y_real)
num_classes = len(classes)

# --- SANITY CHECK: Verify base model accuracies ---
print("\n--- Base Model Sanity Check ---")
print(f"Base 1 (LGBR) Accuracy: {accuracy_score(y_real, y_pred_lgbr):.4f}")
print(f"Base 2 (SGB) Accuracy:  {accuracy_score(y_real, y_pred_sgb):.4f}")
print("-------------------------------\n")

# 1. Stack the raw predictions
X_meta_raw = np.column_stack((y_pred_lgbr, y_pred_sgb))

# 2. One-Hot Encode the base predictions
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_meta_encoded = encoder.fit_transform(X_meta_raw)

# 3. Initialize the Meta-Model 
# (Removed 'multi_class' to fix the deprecation warning)
meta_model = LogisticRegression(
    solver='lbfgs', 
    max_iter=1000, 
    C=1.0, 
    random_state=42
)

print("Training Meta-Model...")

# 4. Direct fit and predict (Matches the exact evaluation style of your DST script)
meta_model.fit(X_meta_encoded, y_real)
y_pred_stack = meta_model.predict(X_meta_encoded)
P_stack = meta_model.predict_proba(X_meta_encoded)

# Output fused predictions
df_fused = pd.DataFrame({
    "y_real": y_real,
    "y_pred_Base1": y_pred_lgbr,
    "y_pred_Base2": y_pred_sgb,
    "y_pred_Stacking": y_pred_stack
})

print("--- Sample Predictions ---")
print(df_fused.head())
# =====================================================================
# 3. GENERATE METRICS & EXPORT
# =====================================================================

print("\nGenerating Classification Reports...")
df_metrics, df_roc, df_cm = build_classification_reports(y_real, y_pred_stack, P_stack)

print("\n--- Metrics Summary ---")
print(df_metrics)

# Exporting everything neatly into an Excel file with multiple sheets
output_excel = "Stacking_MultiClass_Report.xlsx"
with pd.ExcelWriter(output_excel) as writer:
    df_fused.to_excel(writer, sheet_name="Predictions", index=False)
    df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    df_cm.to_excel(writer, sheet_name="Confusion_Matrix", index=True)
    df_roc.to_excel(writer, sheet_name="ROC_AUC", index=False)

print(f"\nAll data successfully exported to {output_excel}")