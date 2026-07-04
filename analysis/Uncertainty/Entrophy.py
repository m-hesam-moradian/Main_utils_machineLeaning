import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

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

    df = None
    for sep in [None, ",", "\t", " "]: 
        try:
            df_temp = pd.read_csv(p, sep=sep, engine='python' if sep is None else 'c')
            if df_temp.shape[1] > 1:
                df = df_temp
                break
        except:
            continue
            
    if df is None:
        try:
            arr = np.loadtxt(p)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            cols = ["y_true", "y_pred"] + [f"prob_class_{i}" for i in range(arr.shape[1]-2)]
            df = pd.DataFrame(arr, columns=cols)
        except Exception:
            raise RuntimeError(f"Could not parse file {p}. Ensure it is CSV or whitespace separated.")

    original_cols = df.columns.tolist()
    cols_map = {c.lower(): c for c in original_cols}
    rename_map = {}
    
    # Map Truth
    if "y_true" in cols_map: rename_map[cols_map["y_true"]] = "y_true"
    elif "label" in cols_map: rename_map[cols_map["label"]] = "y_true"
    elif "target" in cols_map: rename_map[cols_map["target"]] = "y_true"
    else: rename_map[original_cols[0]] = "y_true"

    # Map Pred
    if "y_pred" in cols_map: rename_map[cols_map["y_pred"]] = "y_pred"
    elif "pred" in cols_map: rename_map[cols_map["pred"]] = "y_pred"
    elif "predicted" in cols_map: rename_map[cols_map["predicted"]] = "y_pred"
    else:
        if len(original_cols) > 1:
            rename_map[original_cols[1]] = "y_pred"

    df = df.rename(columns=rename_map)

    current_cols = df.columns
    prob_cols = []
    
    for c in current_cols:
        c_lower = c.lower()
        if c in ["y_true", "y_pred"]: continue
        if "prob" in c_lower:
            prob_cols.append(c)
    
    if not prob_cols and len(current_cols) > 2:
        potential_probs = [c for c in current_cols if c not in ["y_true", "y_pred"]]
        if all(pd.api.types.is_numeric_dtype(df[c]) for c in potential_probs):
            prob_cols = potential_probs

    final_cols = ["y_true"]
    if "y_pred" in df.columns:
        final_cols.append("y_pred")
    final_cols.extend(prob_cols)
    
    return df[final_cols]

# -------------------------
# NORMALIZED Entropy function (0 to 1)
# -------------------------
def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Calculates the normalized entropy (uncertainty) between 0 and 1.
    0 = Perfect certainty (e.g., [1.0, 0.0, 0.0])
    1 = Maximum uncertainty (e.g., [0.33, 0.33, 0.33])
    """
    n_classes = probs.shape[1]
    
    if n_classes <= 1:
        return np.zeros(probs.shape[0])
    
    probs = np.clip(probs, 1e-12, 1.0)
    raw_entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(n_classes)
    
    return raw_entropy / max_entropy

# -------------------------
# CLI / Main Execution
# -------------------------
def main():
    # --- 1. Define paths to your NPT files ---
    # Update these paths to where your actual files are located
    file_model_1 = r"C:\Users\Sam\Desktop\ML\data\model1.npt"
    file_model_2 = r"C:\Users\Sam\Desktop\ML\data\model2.npt"
    
    print("Loading files...")
    df_m1 = load_predictions_auto(file_model_1)
    df_m2 = load_predictions_auto(file_model_2)
    
    # Ensure both files have the same number of rows
    if len(df_m1) != len(df_m2):
        print(f"Warning: Model 1 has {len(df_m1)} rows, Model 2 has {len(df_m2)} rows.")
    
    # --- 2. Extract probability columns ---
    prob_cols_m1 = [c for c in df_m1.columns if c not in ["y_true", "y_pred"]]
    prob_cols_m2 = [c for c in df_m2.columns if c not in ["y_true", "y_pred"]]
    
    probs_m1 = df_m1[prob_cols_m1].values
    probs_m2 = df_m2[prob_cols_m2].values
    
    # --- 3. Calculate Uncertainty (Normalized Entropy) ---
    uncertainty_m1 = normalized_entropy(probs_m1)
    uncertainty_m2 = normalized_entropy(probs_m2)
    
    # --- 4. Create a unified DataFrame for comparison ---
    # Assuming both files evaluate the same dataset in the same order
    comparison_df = pd.DataFrame({
        "y_true": df_m1["y_true"],
        "y_pred_m1": df_m1["y_pred"] if "y_pred" in df_m1.columns else None,
        "y_pred_m2": df_m2["y_pred"] if "y_pred" in df_m2.columns else None,
        "Uncertainty_Model_1": uncertainty_m1,
        "Uncertainty_Model_2": uncertainty_m2
    })
    
    # --- 5. Output results ---
    print("\n✅ Calculated Uncertainty for both models successfully!")
    print("\n--- Preview of Results ---")
    print(comparison_df.head(10))
    
    # Optional: print summary metrics
    print("\n--- Average Uncertainty ---")
    print(f"Model 1 Average Uncertainty: {uncertainty_m1.mean():.4f}")
    print(f"Model 2 Average Uncertainty: {uncertainty_m2.mean():.4f}")
    
    # Copy to clipboard
    try:
        comparison_df.to_clipboard(index=False)
        print("\n📋 Results copied to clipboard! You can paste them into Excel.")
    except Exception as e:
        print("\nCould not copy to clipboard. (Are you running in a headless environment?)")

if __name__ == "__main__":
    main()