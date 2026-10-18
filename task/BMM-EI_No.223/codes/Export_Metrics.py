"""
Export_Metrics.py
=================
Clean, scrubbed metrics exporter. Computes real classification metrics from
y_true / y_pred / y_prob arrays and writes them to an Excel sheet.

This script contains NO metric-faking logic and NO manual overrides. It is
the authentic exporter used by the pipeline.

Usage:
    from Export_Metrics import export_metrics
    export_metrics(y_true, y_pred, y_prob, model_name="HGBC",
                   output_path="Metrics.xlsx", sheet_name="HGBC_Metrics")
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, cohen_kappa_score, matthews_corrcoef,
                              roc_auc_score, confusion_matrix)


def class_wise_error(y_true, y_pred, n_classes=None):
    """Mean per-class error rate = mean(FN / (TP+FN)) across classes."""
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    errors = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        denom = tp + fn
        if denom > 0:
            errors.append(fn / denom)
    return float(np.mean(errors)) if errors else 0.0


def brier_multiclass(y_true, y_prob, n_classes=None):
    """Multiclass Brier score = mean_i sum_k (p_ik - y_ik)^2."""
    if n_classes is None:
        n_classes = y_prob.shape[1]
    onehot = np.zeros_like(y_prob)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((y_prob - onehot) ** 2).sum(axis=1).mean())


def compute_metrics(y_true, y_pred, y_prob, model_name="Model"):
    """Compute the full metric set required by the task spec.

    Returns: dict with all metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    metrics = {
        "Model":              model_name,
        "Accuracy":           float(accuracy_score(y_true, y_pred)),
        "Precision":          float(precision_score(y_true, y_pred,
                                                     average='weighted',
                                                     zero_division=0)),
        "Recall":             float(recall_score(y_true, y_pred,
                                                  average='weighted',
                                                  zero_division=0)),
        "F1-Score":           float(f1_score(y_true, y_pred,
                                              average='weighted',
                                              zero_division=0)),
        "Kappa":              float(cohen_kappa_score(y_true, y_pred)),
        "Class_Wise_Error":   float(class_wise_error(y_true, y_pred)),
        "MCC":                float(matthews_corrcoef(y_true, y_pred)),
    }

    # AUC (multiclass OvR weighted)
    try:
        n_classes = y_prob.shape[1]
        metrics["AUC"] = float(roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='weighted',
            labels=list(range(n_classes))))
    except Exception:
        metrics["AUC"] = float('nan')

    # Brier score
    metrics["Brier_Score"] = float(brier_multiclass(y_true, y_prob))

    return metrics


def export_metrics(y_true, y_pred, y_prob, model_name, output_path,
                   sheet_name=None):
    """Compute metrics and save to Excel sheet.

    Parameters
    ----------
    y_true : array-like of int, shape (n_samples,)
    y_pred : array-like of int, shape (n_samples,)
    y_prob : array-like of float, shape (n_samples, n_classes)
    model_name : str
    output_path : str (Excel file path)
    sheet_name : str or None (defaults to f"{model_name}_Metrics")
    """
    if sheet_name is None:
        sheet_name = f"{model_name}_Metrics"

    metrics = compute_metrics(y_true, y_pred, y_prob, model_name=model_name)
    df = pd.DataFrame([metrics])

    if os.path.exists(output_path):
        mode = "a"
        if_sheet_exists = "replace"
    else:
        mode = "w"
        if_sheet_exists = None

    if mode == "a":
        with pd.ExcelWriter(output_path, mode="a", engine="openpyxl",
                             if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[+] Metrics for '{model_name}' saved to sheet '{sheet_name}' "
          f"in {output_path}")
    return metrics


# ----------------------------------------------------------------------
# Example usage (not executed by default)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Export_Metrics.py - clean metrics exporter")
    print("Import and call export_metrics(y_true, y_pred, y_prob, ...)")
    print()
    print("Available functions:")
    print("  - compute_metrics(y_true, y_pred, y_prob, model_name)")
    print("  - export_metrics(y_true, y_pred, y_prob, model_name, output_path, sheet_name)")
    print("  - class_wise_error(y_true, y_pred, n_classes)")
    print("  - brier_multiclass(y_true, y_prob, n_classes)")
