# metrics_report_column_oriented.py
"""
Column-oriented metrics report generator for label + predicted label + class probabilities files.

Expected input (whitespace or tab separated):
y_true  y_pred  prob_class0  prob_class1  prob_class2  ...

Example (3-class):
0 0 0.6064521149 0.0905436299 0.3030042551
"""

from pathlib import Path
import argparse
import json
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    brier_score_loss
)

# -------------------------
# Loading utilities
# -------------------------
def load_npt_three_class(path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a whitespace/tab/comma separated file where each row is:
      y_true  y_pred  p_class0  p_class1  p_class2

    Returns DataFrame with columns:
      'y_true', 'y_pred', 'prob_class_0', 'prob_class_1', 'prob_class_2'
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    arr = np.loadtxt(p)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # single row
        if arr.size < 5:
            raise ValueError("Expected at least 5 columns for 3-class format (y_true, y_pred, 3 probs).")
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError("Expected format: y_true, y_pred, prob_c0, prob_c1, prob_c2 (>=5 columns)")

    y_true = arr[:, 0].astype(int)
    y_pred = arr[:, 1].astype(int)
    probs = arr[:, 2:5].astype(float)  # exactly 3 class probs

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })
    for i in range(3):
        df[f"prob_class_{i}"] = probs[:, i]

    return df

def load_predictions_auto(path: Union[str, Path]) -> pd.DataFrame:
    """
    Generic loader: tries the 3-class npt format first, then tries CSV/TSV, then generic npy/npz.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    # Try as simple text two-label + probs (3-class) format
    try:
        df = load_npt_three_class(p)
        return df
    except Exception:
        pass

    # try pandas read
    try:
        df = pd.read_csv(p)
        # normalize lower-case column names
        cols_lower = {c.lower(): c for c in df.columns}
        # try find y_true and y_pred
        rename_map = {}
        if "y_true" in cols_lower:
            rename_map[cols_lower["y_true"]] = "y_true"
        elif "y_real" in cols_lower:
            rename_map[cols_lower["y_real"]] = "y_true"
        elif "label" in cols_lower:
            rename_map[cols_lower["label"]] = "y_true"
        if "y_pred" in cols_lower:
            rename_map[cols_lower["y_pred"]] = "y_pred"
        elif "pred" in cols_lower:
            rename_map[cols_lower["pred"]] = "y_pred"

        if rename_map:
            df = df.rename(columns=rename_map)

        # detect probability columns like prob, prob_pos, prob_class_*
        prob_cols = [c for c in df.columns if c.lower().startswith("prob")]
        # If we have no explicit prob columns but numeric columns in [0,1], take them (except y_true/y_pred)
        if not prob_cols:
            for c in df.columns:
                if c in ("y_true", "y_pred"):
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    s = df[c].dropna()
                    if len(s) > 0 and s.between(0, 1).all():
                        prob_cols.append(c)
        # if we found probability columns, keep them
        if "y_true" in df.columns and "y_pred" in df.columns and prob_cols:
            return df[["y_true", "y_pred"] + prob_cols]

        # fallback: if there are at least 5 columns, assume format y_true, y_pred, p0,p1,p2
        if df.shape[1] >= 5:
            cols = df.columns.tolist()
            return df.rename(columns={cols[0]: "y_true", cols[1]: "y_pred"})[["y_true", "y_pred", cols[2], cols[3], cols[4]]]
    except Exception:
        pass

    # try numpy load (npy or npz)
    try:
        import numpy as _np
        if p.suffix.lower() == ".npy":
            arr = _np.load(p, allow_pickle=True)
            arr = _np.asarray(arr)
            if arr.ndim == 1 and arr.size >= 5:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] >= 5:
                # assume first two are labels, next three probs
                y_true = arr[:, 0].astype(int)
                y_pred = arr[:, 1].astype(int)
                probs = arr[:, 2:5].astype(float)
                df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
                for i in range(3):
                    df[f"prob_class_{i}"] = probs[:, i]
                return df
    except Exception:
        pass

    raise RuntimeError("Failed to auto-load the predictions file. Please ensure it matches the expected format.")

# -------------------------
# Brier score & decomposition
# -------------------------
def brier_score_multiclass(y_true: np.ndarray, prob_matrix: np.ndarray, labels: List) -> float:
    """
    Multiclass Brier score:
      BS = mean_i sum_k (p_{ik} - y_{ik})^2
    """
    p = np.asarray(prob_matrix, dtype=float)
    n, K = p.shape
    # build one-hot for y_true according to labels
    label_to_idx = {str(l): i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx.get(str(v), None) for v in y_true])
    if (y_idx == None).any():
        raise ValueError("y_true contains labels not found in provided labels list.")
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(n), y_idx] = 1.0
    diffsq = (p - y_onehot) ** 2
    return float(diffsq.sum(axis=1).mean())

def brier_decomposition_binary(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10, strategy: str = "quantile") -> Dict[str, float]:
    """
    Decompose binary Brier: returns brier, reliability, resolution, uncertainty.
    """
    y = np.asarray(y_true).astype(float)
    p = np.asarray(prob_pos).astype(float)
    N = len(y)
    if strategy == "quantile":
        try:
            bins = pd.qcut(p, q=n_bins, duplicates="drop")
        except Exception:
            bins = pd.cut(p, bins=n_bins)
    else:
        bins = pd.cut(p, bins=n_bins)
    df = pd.DataFrame({"y": y, "p": p, "bin": bins})
    o = df["y"].mean()
    reliability = 0.0
    resolution = 0.0
    for _, g in df.groupby("bin"):
        ng = len(g)
        if ng == 0:
            continue
        p_g = g["p"].mean()
        o_g = g["y"].mean()
        reliability += (ng / N) * ((o_g - p_g) ** 2)
        resolution += (ng / N) * ((o_g - o) ** 2)
    uncertainty = o * (1.0 - o)
    brier = reliability - resolution + uncertainty
    return {"brier": float(brier), "reliability": float(reliability), "resolution": float(resolution), "uncertainty": float(uncertainty)}

def brier_decomposition_multiclass(y_true: np.ndarray, prob_matrix: np.ndarray, labels: List[int], n_bins: int = 10, strategy: str = "quantile") -> Dict[str, Dict[str, float]]:
    """
    Compute one-vs-rest binary decomposition for each class.
    Returns dict: {label: decomposition_dict}
    """
    res = {}
    for i, lab in enumerate(labels):
        p_k = prob_matrix[:, i]
        y_k = (np.array(y_true) == lab).astype(float)
        res[lab] = brier_decomposition_binary(y_k, p_k, n_bins=n_bins, strategy=strategy)
    return res

# -------------------------
# Additional utilities
# -------------------------
def compute_markedness(y_true, y_pred) -> Optional[float]:
    try:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        precs = []
        npvs = []
        for cls in labels:
            y_t = (np.array(y_true) == cls).astype(int)
            y_p = (np.array(y_pred) == cls).astype(int)
            prec = precision_score(y_t, y_p, zero_division=0)
            cm = confusion_matrix(y_t, y_p)
            # compute TN and FN robustly
            if cm.size == 1:
                tn = int(cm[0, 0]); fn = 0
            else:
                # Try to interpret as [[tn, fp],[fn, tp]]
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
                elif cm.shape[0] == 1:
                    tn = int(cm[0,0]); fn = 0
                else:
                    # fallback
                    tn = int(cm[0,0]); fn = 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            precs.append(prec); npvs.append(npv)
        return float(np.mean(precs) + np.mean(npvs) - 1.0)
    except Exception:
        return None

# -------------------------
# Report builder
# -------------------------
def build_column_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a column-oriented report DataFrame with Metric / Value pairs.

    df must contain:
      - y_true
      - optionally y_pred
      - probability columns named 'prob_class_0', 'prob_class_1', 'prob_class_2' for 3 classes
    """
    if "y_true" not in df.columns:
        raise ValueError("df must contain 'y_true' column")
    y_true = df["y_true"].values
    has_pred = "y_pred" in df.columns
    y_pred = df["y_pred"].values if has_pred else None

    # detect prob columns for 3-class
    prob_cols = [f"prob_class_{i}" for i in range(3) if f"prob_class_{i}" in df.columns]
    if len(prob_cols) != 3:
        # try alternative names or numeric columns
        numeric_cols = [c for c in df.columns if c not in ("y_true", "y_pred") and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 3:
            prob_cols = numeric_cols[:3]
        else:
            prob_cols = []

    prob_matrix = None
    if len(prob_cols) == 3:
        prob_matrix = df[prob_cols].values.astype(float)

    metrics = []
    add = metrics.append

    # Basic
    add(("N", int(len(df))))
    unique_labels = np.unique(y_true)
    add(("Labels", ",".join([str(x) for x in unique_labels])))

    # Classification metrics
    if has_pred:
        acc = accuracy_score(y_true, y_pred)
        add(("Accuracy", round(float(acc), 6)))
        add(("Precision_macro", round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
        add(("Recall_macro", round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
        add(("F1_macro", round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
        add(("Precision_micro", round(float(precision_score(y_true, y_pred, average="micro", zero_division=0)), 6)))
        add(("Recall_micro", round(float(recall_score(y_true, y_pred, average="micro", zero_division=0)), 6)))
        add(("F1_micro", round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 6)))
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
            add(("MCC", round(float(mcc), 6)))
        except Exception:
            add(("MCC", None))

        # per-class metrics using union of seen labels in y_true and y_pred
        labels_for_metrics = np.unique(np.concatenate([unique_labels, np.unique(y_pred)]))
        precs = precision_score(y_true, y_pred, average=None, labels=labels_for_metrics, zero_division=0)
        recs = recall_score(y_true, y_pred, average=None, labels=labels_for_metrics, zero_division=0)
        f1s = f1_score(y_true, y_pred, average=None, labels=labels_for_metrics, zero_division=0)
        for lab, p, r, f in zip(labels_for_metrics, precs, recs, f1s):
            add((f"Class_{lab}_Precision", round(float(p), 6)))
            add((f"Class_{lab}_Recall", round(float(r), 6)))
            add((f"Class_{lab}_F1", round(float(f), 6)))
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels_for_metrics)
        add(("ConfusionMatrix_shape", f"{cm.shape[0]}x{cm.shape[1]}"))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                add((f"CM[{i},{j}]", int(cm[i, j])))
    else:
        add(("Accuracy", None))
        add(("Precision_macro", None))
        add(("Recall_macro", None))
        add(("F1_macro", None))

    # AUC (multiclass OVR) if probabilities available
    if prob_matrix is not None:
        # labels order for prob_matrix — assume class order 0,1,2
        prob_labels = [0, 1, 2]
        try:
            # Map y_true to int indices if needed
            y_true_int = np.array([int(x) for x in y_true])
            auc_ovr = roc_auc_score(y_true_int, prob_matrix, multi_class="ovr")
            add(("AUC_ovr_multiclass", round(float(auc_ovr), 6)))
        except Exception:
            add(("AUC_ovr_multiclass", None))

        # Multiclass Brier
        try:
            bs = brier_score_multiclass(y_true, prob_matrix, labels=prob_labels)
            add(("Brier_multiclass", round(float(bs), 8)))
            # Brier Skill Score (ref: class prevalences)
            prevalences = np.array([(np.array(y_true) == lab).mean() for lab in prob_labels], dtype=float)
            prob_ref_mat = np.tile(prevalences, (len(y_true), 1))
            # compute reference BS
            bs_ref = brier_score_multiclass(y_true, prob_ref_mat, labels=prob_labels)
            bss = None if bs_ref == 0 else 1.0 - (bs / bs_ref)
            add(("BrierSkillScore_multiclass", round(float(bss), 6) if bss is not None else None))
        except Exception as e:
            add(("Brier_multiclass_error", str(e)))

        # Per-class decomposition
        try:
            decomp = brier_decomposition_multiclass(y_true, prob_matrix, labels=prob_labels, n_bins=10, strategy="quantile")
            for lab, comp in decomp.items():
                add((f"Brier_decomp_{lab}_brier", round(float(comp["brier"]), 8)))
                add((f"Brier_decomp_{lab}_reliability", round(float(comp["reliability"]), 8)))
                add((f"Brier_decomp_{lab}_resolution", round(float(comp["resolution"]), 8)))
                add((f"Brier_decomp_{lab}_uncertainty", round(float(comp["uncertainty"]), 8)))
        except Exception as e:
            add(("Brier_decomp_error", str(e)))
    else:
        add(("AUC_ovr_multiclass", None))
        add(("Brier_multiclass", None))
        add(("BrierSkillScore_multiclass", None))

    # Markedness
    if has_pred:
        marked = compute_markedness(y_true, y_pred)
        add(("Markedness", round(float(marked), 6) if marked is not None else None))
    else:
        add(("Markedness", None))

    # Create DataFrame
    df_rows = pd.DataFrame(metrics, columns=["Metric", "Value"])
    df_rows.insert(0, "No.", range(1, len(df_rows) + 1))
    return df_rows

# -------------------------
# Save/export
# -------------------------
def save_and_copy(df_rows: pd.DataFrame, out_path: Union[str, Path] = "metrics_column_report.xlsx") -> Path:
    p = Path(out_path)
    try:
        df_rows.to_excel(p, index=False)
        saved = p
    except Exception:
        csvp = p.with_suffix(".csv")
        df_rows.to_csv(csvp, index=False)
        saved = csvp
    # clipboard best-effort
    try:
        df_rows.to_clipboard(index=False)
    except Exception:
        pass
    return saved

# -------------------------
# CLI / Example run
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate a column-oriented metrics report from predictions.")
    parser.add_argument("preds", help="Path to predictions file (your .npt with 3-class probabilities works).")
    parser.add_argument("--out", "-o", default="metrics_column_report.xlsx", help="Output Excel/CSV path")
    args = parser.parse_args()

    df_in = load_predictions_auto(args.preds)
    report = build_column_report(df_in)
    saved = save_and_copy(report, args.out)
    print("✅ Column-oriented report created.")
    print("Saved to:", saved)
    print("\nPreview:")
    print(report.to_string(index=False))

if __name__ == "__main__":
    main()
