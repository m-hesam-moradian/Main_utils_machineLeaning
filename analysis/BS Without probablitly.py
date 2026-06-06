"""
Metrics report generator using only Truth and Prediction columns.
If probabilities are missing, it treats predictions as 100% confidence for Brier calculations.
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# -------------------------
# Loading utilities
# -------------------------
def load_predictions_simple(path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads file and ensures y_true and y_pred exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    df = None
    for sep in [None, ",", "\t", " "]: 
        try:
            df_temp = pd.read_csv(p, sep=sep, engine='python' if sep is None else 'c')
            if df_temp.shape[1] >= 2:
                df = df_temp
                break
        except:
            continue
            
    if df is None:
        arr = np.loadtxt(p)
        df = pd.DataFrame(arr[:, :2], columns=["y_true", "y_pred"])

    # Normalize column names
    original_cols = df.columns.tolist()
    cols_map = {c.lower(): c for c in original_cols}
    rename_map = {}
    
    # Map Truth
    for key in ["y_true", "label", "target", "actual"]:
        if key in cols_map:
            rename_map[cols_map[key]] = "y_true"
            break
    if "y_true" not in rename_map: rename_map[original_cols[0]] = "y_true"

    # Map Pred
    for key in ["y_pred", "pred", "predicted"]:
        if key in cols_map:
            rename_map[cols_map[key]] = "y_pred"
            break
    if "y_pred" not in rename_map and len(original_cols) > 1: 
        rename_map[original_cols[1]] = "y_pred"

    df = df.rename(columns=rename_map)
    return df[["y_true", "y_pred"]]

# -------------------------
# Brier logic for Hard Labels
# -------------------------
def brier_score_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Brier Score using hard predictions (0 or 1).
    BS = 1/N * sum((y_true_onehot - y_pred_onehot)^2)
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)
    n_samples = len(y_true)
    
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Create one-hot versions
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_pred_onehot = np.zeros((n_samples, n_classes))
    
    for i in range(n_samples):
        y_true_onehot[i, label_to_idx[y_true[i]]] = 1.0
        y_pred_onehot[i, label_to_idx[y_pred[i]]] = 1.0
        
    # Brier formula: mean of squared differences across all classes
    mse = np.mean(np.sum((y_true_onehot - y_pred_onehot)**2, axis=1))
    return float(mse)

# -------------------------
# Report builder
# -------------------------
def build_simple_report(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    # Clean data (handle strings vs ints)
    try:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
    except:
        y_true = y_true.astype(str)
        y_pred = y_pred.astype(str)

    metrics = []
    add = metrics.append

    # Basic Info
    add(("N Samples", int(len(df))))
    unique_labels = np.unique(y_true)
    add(("Class Labels", ", ".join([str(x) for x in unique_labels])))

    # Standard Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    add(("Accuracy", round(float(acc), 6)))
    
    add(("Precision (Macro)", round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    add(("Recall (Macro)", round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    add(("F1 Score (Macro)", round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6)))

    # Brier Score (calculated from hard labels)
    try:
        bs = brier_score_from_labels(y_true, y_pred)
        add(("Brier Score (Hard)", round(bs, 6)))
        # Note: With hard labels, Brier Score is essentially 2 * (1 - Accuracy) in multiclass
    except Exception as e:
        add(("Brier Score Error", str(e)))

    # Create Final DataFrame
    res_df = pd.DataFrame(metrics, columns=["Metric", "Value"])
    res_df.insert(0, "No.", range(1, len(res_df) + 1))
    return res_df

# -------------------------
# Main Execution
# -------------------------
def main():
    # Update this path to your file
    input_file = r"C:\Users\Sam\Desktop\ML\data\Data_err.npt"
    
    try:
        df_in = load_predictions_simple(input_file)
        report = build_simple_report(df_in)
        
        # Copy to clipboard for Excel/Word
        report.to_clipboard(index=False)

        print("✅ Report generated from Truth and Prediction columns.")
        print(report.to_string(index=False))
        print("\n*Note: Brier Score calculated using hard predictions (Prob=1.0 for predicted class).")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()