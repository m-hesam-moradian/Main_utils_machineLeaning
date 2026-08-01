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
    # -------------------------
    # Define model files
    # -------------------------
    model_files = {
        "ETC(CHI2)": r"C:\Users\Sam\Desktop\ML\data\model1.npt",
        "ETC(CHI2) + SEOA": r"C:\Users\Sam\Desktop\ML\data\model2.npt",
        "ETC(CHI2) + POA": r"C:\Users\Sam\Desktop\ML\data\model3.npt",
        "ETC(RFE)": r"C:\Users\Sam\Desktop\ML\data\model3.npt",
        "ETC(RFE) + SEOA": r"C:\Users\Sam\Desktop\ML\data\model3.npt",
        "ETC(RFE) + POA": r"C:\Users\Sam\Desktop\ML\data\model3.npt",
    }

    print("Loading files...")

    dataframes = {}
    uncertainties = {}

    # -------------------------
    # Load each model
    # -------------------------
    for model_name, file_path in model_files.items():
        df = load_predictions_auto(file_path)
        dataframes[model_name] = df

        prob_cols = [c for c in df.columns if c not in ["y_true", "y_pred"]]
        probs = df[prob_cols].values

        uncertainties[model_name] = normalized_entropy(probs)

    # -------------------------
    # Check lengths
    # -------------------------
    lengths = [len(df) for df in dataframes.values()]
    if len(set(lengths)) != 1:
        print("Warning: Models have different numbers of samples.")

    # -------------------------
    # Create comparison table
    # -------------------------
    first_model = list(model_files.keys())[0]

    comparison_df = pd.DataFrame({
        "y_true": dataframes[first_model]["y_true"]
    })

    for model_name in model_files.keys():
        if "y_pred" in dataframes[model_name]:
            comparison_df[f"y_pred_{model_name}"] = dataframes[model_name]["y_pred"]

        comparison_df[f"Uncertainty_{model_name}"] = uncertainties[model_name]

    # -------------------------
    # Show results
    # -------------------------
    print("\nPreview:")
    print(comparison_df.head(10))

    print("\nAverage Uncertainty")
    for model_name in model_files.keys():
        print(f"{model_name}: {uncertainties[model_name].mean():.4f}")

    # Copy to clipboard
    try:
        comparison_df.to_clipboard(index=False)
        print("\nResults copied to clipboard.")
    except:
        print("\nCould not copy to clipboard.")
if __name__ == "__main__":
    main()