# metrics_report_column_oriented.py
"""
Column-oriented metrics report generator for label + predicted label + class probabilities files.

Expected input (whitespace or tab separated):
y_true  y_pred  prob_class0  prob_class1  prob_class2  ...

Example (3-class):
0 0 0.6064521149 0.0905436299 0.30300425511
"""

from pathlib import Path

from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,
   
)

# -------------------------
# Loading utilities
# -------------------------
def load_predictions_auto(path: Union[str, Path]) -> pd.DataFrame:
    """
    Generic loader: 
    1. Tries to read as standard CSV/TSV/Whitespace.
    2. Auto-detects 'y_true', 'y_pred'.
    3. Auto-detects ALL probability columns (prob_class_X).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    # Attempt to load using pandas with various separators
    df = None
    # Try common separators
    for sep in [None, ",", "\t", " "]: 
        try:
            # sep=None causes pandas to use the python sniffer, which is good for whitespace
            df_temp = pd.read_csv(p, sep=sep, engine='python' if sep is None else 'c')
            if df_temp.shape[1] > 1:
                df = df_temp
                break
        except:
            continue
            
    # Fallback: simple numpy load if pandas failed completely
    if df is None:
        try:
            arr = np.loadtxt(p)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            # Default headers if no header found
            cols = ["y_true", "y_pred"] + [f"prob_class_{i}" for i in range(arr.shape[1]-2)]
            df = pd.DataFrame(arr, columns=cols)
        except Exception:
            raise RuntimeError("Could not parse file. Ensure it is CSV or whitespace separated.")

    # 1. Normalize Column Names to lower case for searching
    original_cols = df.columns.tolist()
    cols_map = {c.lower(): c for c in original_cols}

    # 2. Find y_true / y_pred
    rename_map = {}
    
    # Map Truth
    if "y_true" in cols_map: rename_map[cols_map["y_true"]] = "y_true"
    elif "label" in cols_map: rename_map[cols_map["label"]] = "y_true"
    elif "target" in cols_map: rename_map[cols_map["target"]] = "y_true"
    else: 
        # Fallback: assume 1st column is True
        rename_map[original_cols[0]] = "y_true"

    # Map Pred
    if "y_pred" in cols_map: rename_map[cols_map["y_pred"]] = "y_pred"
    elif "pred" in cols_map: rename_map[cols_map["pred"]] = "y_pred"
    elif "predicted" in cols_map: rename_map[cols_map["predicted"]] = "y_pred"
    else:
        # Fallback: assume 2nd column is Pred
        if len(original_cols) > 1:
            rename_map[original_cols[1]] = "y_pred"

    df = df.rename(columns=rename_map)

    # 3. Dynamic Probability Column Detection
    # We look for columns that started with 'prob' (case insensitive) 
    # OR columns that look like prob_class_X
    current_cols = df.columns
    prob_cols = []
    
    for c in current_cols:
        c_lower = c.lower()
        if c in ["y_true", "y_pred"]: continue
        
        # Criteria: name contains 'prob' or is purely numeric 0-1 (optional heuristic)
        if "prob" in c_lower:
            prob_cols.append(c)
    
    # If no named prob cols found, look for remaining numeric columns
    if not prob_cols and len(current_cols) > 2:
        potential_probs = [c for c in current_cols if c not in ["y_true", "y_pred"]]
        # Simple check: take them if they are numeric
        if all(pd.api.types.is_numeric_dtype(df[c]) for c in potential_probs):
            prob_cols = potential_probs

    # Return only relevant columns
    final_cols = ["y_true"]
    if "y_pred" in df.columns:
        final_cols.append("y_pred")
    final_cols.extend(prob_cols)
    
    return df[final_cols]

# -------------------------
# Excel multi-model loader
# -------------------------
def load_models_from_excel(path, sheet_name="Probs"):

    df = pd.read_excel(
        path,
        sheet_name=sheet_name
    )

    columns = df.columns.tolist()

    # Find every model beginning
    start_cols = [
        i for i, c in enumerate(columns)
        if str(c).endswith("_y_real")
    ]

    models = {}

    for idx, start in enumerate(start_cols):

        model_name = (
            str(columns[start])
            .replace("_y_real", "")
        )

        # Define block end
        if idx + 1 < len(start_cols):
            end = start_cols[idx + 1]
        else:
            end = len(columns)


        block = df.iloc[:, start:end].copy()


        rename = {}

        for col in block.columns:

            col = str(col)

            if col.endswith("_y_real"):
                rename[col] = "y_true"

            elif col.endswith("_y_pred"):
                rename[col] = "y_pred"

            elif "prob_" in col.lower():
                rename[col] = col


        block = block.rename(
            columns=rename
        )


        # Remove empty rows
        block = block.dropna(
            subset=["y_true"]
        ).reset_index(drop=True)


        models[model_name] = block


    return models
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
    if "y_true" not in df.columns:
        raise ValueError("df must contain 'y_true' column")
    
    y_true = df["y_true"].values
    # Ensure y_true are standard integers if they look like numbers
    try:
        y_true = y_true.astype(int)
    except:
        pass # keep as strings if they are strings

    has_pred = "y_pred" in df.columns
    y_pred = df["y_pred"].values if has_pred else None
    if has_pred:
        try: y_pred = y_pred.astype(int)
        except: pass

    # --- DYNAMIC PROBABILITY DETECTION ---
    # Identify probability columns
    prob_cols = [c for c in df.columns if c not in ["y_true", "y_pred"]]
    
    # Sort them to ensure logical order. 
    # If they are named "prob_class_0", "prob_class_1", sorting works naturally.
    # We try to extract the integer suffix to sort correctly (10 comes after 2)
    def natural_keys(text):
        import re
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
    
    prob_cols.sort(key=natural_keys)

    prob_matrix = None
    prob_labels = []

    if len(prob_cols) > 0:
        prob_matrix = df[prob_cols].values.astype(float)
        
        # --- DYNAMIC LABEL EXTRACTION ---
        # Try to infer label from column name (e.g., "prob_class_5" -> 5)
        # If columns are just "prob0", "prob1", we assume labels are 0, 1...
        # If columns are generic "col3", "col4", we assume they map to sorted(unique(y_true))
        
        inferred_labels = []
        is_generic_names = True
        
        for col in prob_cols:
            # Look for number at the end of the string
            import re
            match = re.search(r'(\d+)$', col)
            if match and "prob" in col.lower():
                inferred_labels.append(int(match.group(1)))
                is_generic_names = False
            else:
                inferred_labels.append(col)

        if not is_generic_names and len(set(inferred_labels)) == len(prob_cols):
            # We successfully extracted IDs like 0, 1, 2 from names
            prob_labels = inferred_labels
        else:
            # Fallback: Assume the probability columns correspond to the sorted unique classes in data
            unique_classes = sorted(np.unique(y_true))
            # If we have more probability columns than classes in y_true (e.g. class 2 never appears),
            # we assume the range 0..N-1
            if len(prob_cols) == len(unique_classes):
                prob_labels = unique_classes
            else:
                prob_labels = list(range(len(prob_cols)))

    # -------------------------------------

    metrics = []
    add = metrics.append

    # Basic
    add(("N", int(len(df))))
    unique_labels = np.unique(y_true)
    add(("Labels", ",".join([str(x) for x in unique_labels])))

    # Classification metrics
    # if has_pred:
    #     acc = accuracy_score(y_true, y_pred)
    #     add(("Accuracy", round(float(acc), 6)))
    #     add(("Precision_macro", round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    #     add(("Recall_macro", round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    #     add(("F1_macro", round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6)))
    # else:
    #     add(("Accuracy", None))

    # Multiclass Brier and AUC
    if prob_matrix is not None:
        # try:
        #     # AUC
        #     # We need to map y_true to indices matching prob_matrix columns for AUC
        #     # Simple heuristic: if classes are 0..K-1, works out of box.
        #     auc_ovr = roc_auc_score(y_true, prob_matrix, multi_class="ovr", average="macro", labels=prob_labels)
        #     add(("AUC_ovr_multiclass", round(float(auc_ovr), 6)))
        # except Exception:
        #     add(("AUC_ovr_multiclass", None))

        # Brier
        try:
            # Pass the dynamically detected prob_labels
            bs = brier_score_multiclass(y_true, prob_matrix, labels=prob_labels)
            add(("Brier_multiclass", round(float(bs), 8)))
        except Exception as e:
            add(("Brier_multiclass_error", str(e)))

        # Decomposition
        try:
            # Pass the dynamically detected prob_labels
            decomp = brier_decomposition_multiclass(y_true, prob_matrix, labels=prob_labels, n_bins=10)
            for lab, comp in decomp.items():
                add((f"Brier_decomp_{lab}_brier", round(float(comp["brier"]), 8)))
                add((f"Brier_decomp_{lab}_reliability", round(float(comp["reliability"]), 8)))
                add((f"Brier_decomp_{lab}_resolution", round(float(comp["resolution"]), 8)))
                add((f"Brier_decomp_{lab}_uncertainty", round(float(comp["uncertainty"]), 8)))
        except Exception as e:
            add(("Brier_decomp_error", str(e)))

    # Create DataFrame
    df_rows = pd.DataFrame(metrics, columns=["Metric", "Value"])
    df_rows.insert(0, "No.", range(1, len(df_rows) + 1))
    return df_rows

# -------------------------
# CLI / Example run
# -------------------------
# -------------------------
# CLI / Excel Example run
# -------------------------
def main():

    excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"


    print("Loading Excel probability sheet...")


    models = load_models_from_excel(
        excel_path,
        sheet_name="Probs(ENN)"
    )


    print("\nDetected models:")

    for model in models:
        print(" -", model)



    all_reports = []


    for model_name, df_model in models.items():

        print(
            f"\nProcessing {model_name}"
        )


        report = build_column_report(
            df_model
        )


        # Add model name column
        report.insert(
            0,
            "Model",
            model_name
        )


        all_reports.append(
            report
        )


    final_report = pd.concat(
        all_reports,
        ignore_index=True
    )


    final_report.to_clipboard(
        index=False
    )


    print(
        "\n✅ All model reports copied to clipboard."
    )


    print(
        final_report.to_string(index=False)
    )


if __name__ == "__main__":
    main()