"""
Export_Metrics.py
=================
Clean, scrubbed metrics exporter. Computes real classification metrics from
y_true / y_pred / y_prob arrays and returns them as a DataFrame.

NO metric-faking logic. NO manual overrides.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, brier_score_loss)


def class_wise_error(y_true, y_pred, n_classes=None):
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    errors = []
    for i in range(n_classes):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp
        denom = tp + fn
        if denom > 0: errors.append(fn / denom)
    return float(np.mean(errors)) if errors else 0.0


def compute_metrics(y_true, y_pred, y_prob, model_name="Model"):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    m = {"Model": model_name,
         "Accuracy": float(accuracy_score(y_true, y_pred)),
         "Precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
         "Recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
         "F1-Score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
         "Kappa": float(cohen_kappa_score(y_true, y_pred)),
         "MCC": float(matthews_corrcoef(y_true, y_pred)),
         "Class_Wise_Error": float(class_wise_error(y_true, y_pred))}
    try:
        n = y_prob.shape[1]
        m["AUC"] = float(roc_auc_score(y_true, y_prob, multi_class='ovr',
                         average='weighted', labels=list(range(n))))
    except: m["AUC"] = float('nan')
    onehot = np.zeros_like(y_prob)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    m["Brier_Score"] = float(((y_prob - onehot) ** 2).sum(axis=1).mean())
    return m


if __name__ == "__main__":
    print("Export_Metrics.py - clean metrics exporter")
    print("Import and call: compute_metrics(y_true, y_pred, y_prob, model_name)")
