# metrics_report_column_oriented.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)

# -------------------------
# Utility functions
# -------------------------
def markedness(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precs = []
    npvs = []
    for cls in labels:
        y_t = (np.array(y_true) == cls).astype(int)
        y_p = (np.array(y_pred) == cls).astype(int)
        prec = precision_score(y_t, y_p, zero_division=0)
        cm = confusion_matrix(y_t, y_p)
        if cm.shape == (1, 1):
            tn = cm[0, 0]; fn = 0
        else:
            tn = cm[0, 0]; fn = cm[1, 0] if cm.shape[0] > 1 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        precs.append(prec); npvs.append(npv)
    return float(np.mean(precs) + np.mean(npvs) - 1.0)

def class_wise_error(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    errors = {}
    for i, cls in enumerate(labels):
        total = int(cm[i, :].sum())
        correct = int(cm[i, i])
        errors[str(cls)] = None if total == 0 else 1.0 - (correct / total)
    return errors

def per_class_metrics(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {str(lbl): {"precision": float(p), "recall": float(r), "f1": float(f)}
            for lbl, p, r, f in zip(labels, prec, rec, f1)}

# -------------------------
# Load predictions (robust)
# -------------------------
def load_predictions(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Try numpy binary .npy first
    if p.suffix.lower() == ".npy":
        data = np.load(p)
    else:
        # try text load; handle whitespace or comma separated
        try:
            data = np.loadtxt(p)
        except Exception:
            # try pandas read_csv
            df = pd.read_csv(p)
            if {"y_real", "y_pred"}.issubset(df.columns):
                return df[["y_real", "y_pred"]].reset_index(drop=True)
            # try first two columns
            return pd.DataFrame({"y_real": df.iloc[:, 0].values, "y_pred": df.iloc[:, 1].values})
    # If numpy array loaded, ensure shape (n,2)
    data = np.asarray(data)
    if data.ndim == 1 and data.size == 2:
        data = data.reshape(1, 2)
    if data.ndim == 2 and data.shape[1] >= 2:
        y_real = data[:, 0]
        y_pred = data[:, 1]
        return pd.DataFrame({"y_real": y_real, "y_pred": y_pred})
    raise ValueError("Loaded data has unexpected shape; expected two columns (y_real, y_pred).")

# -------------------------
# Build column-oriented report
# -------------------------
def build_column_report(df_preds):
    if not {"y_real", "y_pred"}.issubset(df_preds.columns):
        raise ValueError("Input DataFrame must contain 'y_real' and 'y_pred' columns")

    y_true = df_preds["y_real"].values
    y_pred = df_preds["y_pred"].values
    labels = np.unique(np.concatenate([y_true, y_pred]))
    labels_str = [str(l) for l in labels]

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    marked = markedness(y_true, y_pred)
    cwe = class_wise_error(y_true, y_pred)
    per_class = per_class_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Build list of (metric, value) rows
    rows = []
    add = rows.append

    # Basic summary
    add(("N", len(df_preds)))
    add(("Labels", ",".join(labels_str)))
    add(("Accuracy", round(float(acc), 6)))
    add(("Precision_macro", round(float(prec_macro), 6)))
    add(("Recall_macro", round(float(rec_macro), 6)))
    add(("F1_macro", round(float(f1_macro), 6)))
    add(("MCC", round(float(mcc), 6)))
    add(("Markedness", round(float(marked), 6)))
    add(("AUC_macro", None))            # requires probabilities
    add(("Brier_multiclass", None))     # requires probabilities

    # Confusion matrix rows (one row per cell)
    add(("ConfusionMatrix_shape", f"{cm.shape[0]}x{cm.shape[1]}"))
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            add((f"CM[{r},{c}]", int(cm[r, c])))

    # Class-wise error rows
    for lab in labels_str:
        add((f"CWE_class_{lab}", cwe.get(lab, None)))

    # Per-class metrics rows
    for lab in labels_str:
        metrics = per_class.get(lab, {})
        add((f"Class_{lab}_Precision", metrics.get("precision", None)))
        add((f"Class_{lab}_Recall", metrics.get("recall", None)))
        add((f"Class_{lab}_F1", metrics.get("f1", None)))

    # Create DataFrame with numbered index
    df_rows = pd.DataFrame(rows, columns=["Metric", "Value"])
    df_rows.insert(0, "No.", range(1, len(df_rows) + 1))

    return df_rows

# -------------------------
# Save and copy
# -------------------------
def save_and_copy(df_rows, out_excel="metrics_column_report.xlsx"):
    out_path = Path(out_excel)
    try:
        df_rows.to_excel(out_path, index=False)
    except Exception:
        # fallback to CSV
        out_csv = out_path.with_suffix(".csv")
        df_rows.to_csv(out_csv, index=False)
        out_path = out_csv
    # copy to clipboard
    df_rows.to_clipboard(index=False)
    return out_path

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Update this path to your predictions file (supports .npy, .txt, .csv)
    preds_path = r"C:\Users\Sam\Desktop\ML\data\Data_err.npt"

    # Load predictions
    try:
        df_preds = load_predictions(preds_path)
    except Exception as e:
        # try forgiving load for nonstandard extension
        try:
            arr = np.loadtxt(preds_path)
            df_preds = pd.DataFrame({"y_real": arr[:, 0], "y_pred": arr[:, 1]})
        except Exception as e2:
            raise RuntimeError(f"Failed to load predictions: {e}; fallback error: {e2}")

    # Build column-oriented report
    df_report = build_column_report(df_preds)

    # Save and copy to clipboard
    saved = save_and_copy(df_report, out_excel="metrics_column_report.xlsx")

    print("✅ Column-oriented report created and copied to clipboard.")
    print("Saved to:", saved)
    print("\nPreview:")
    print(df_report.head(40).to_string(index=False))