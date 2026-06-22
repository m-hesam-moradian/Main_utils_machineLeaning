import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# -------------------------
# USER SETTINGS
# -------------------------
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
SHEET_NAME = "predicts"


# -------------------------
# Brier logic for Hard Labels (Kept exactly as requested)
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
    except Exception as e:
        add(("Brier Score Error", str(e)))

    # Create DataFrame (without "No." column, since we will add "Model" name)
    res_df = pd.DataFrame(metrics, columns=["Metric", "Value"])
    return res_df


# -------------------------
# Main Execution (Excel Looping)
# -------------------------
def main():
    try:
        # Load Excel Data
        df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=0)
        columns = df.columns.tolist()
        
        all_reports = []

        # Loop through columns in steps of 2 (Real, Pred)
        for i in range(0, len(columns), 2):
            model_name = columns[i].strip()
            
            # Extract Real and Pred, drop missing values
            temp_df = df.iloc[:, [i, i+1]].dropna()
            
            if temp_df.empty:
                print(f"⚠️ No valid data for {model_name}, skipping...")
                continue
                
            # Rename columns so the report builder recognizes them
            temp_df.columns = ["y_true", "y_pred"]
            
            # Build the metric report for this model
            report = build_simple_report(temp_df)
            
            # Add model name column to the far left
            report.insert(0, "Model", model_name)
            all_reports.append(report)

        # Combine all models into a single DataFrame
        final_report = pd.concat(all_reports, ignore_index=True)

        # Output to Console and Clipboard
        print("\nFinal Metrics Table:\n")
        print(final_report.to_string(index=False))
        print("\n*Note: Brier Score calculated using hard predictions (Prob=1.0 for predicted class).")
        
        final_report.to_clipboard(index=False)
        print("\n✅ Table copied to clipboard — ready to paste into Excel or Word.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()