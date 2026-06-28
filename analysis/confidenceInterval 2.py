import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, matthews_corrcoef,
    confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

def build_ci_classification_reports(y_real, y_pred, model_name="Model", n_bootstrap=1000, ci=95):
    classes = np.unique(y_real)
    
    # Helper 1: Calculate markedness for a given array
    def calculate_markedness(y_t, y_p):
        cm = confusion_matrix(y_t, y_p, labels=classes)
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

    # Helper 2: Calculate all core metrics for a slice of data
    def get_core_metrics(y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        return {
            "Accuracy": acc,
            "Precision": precision_score(y_t, y_p, average="macro", zero_division=0),
            "Recall": recall_score(y_t, y_p, average="macro", zero_division=0),
            "F1": f1_score(y_t, y_p, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_t, y_p),
            "Class-Wise Error": 1 - acc,
            "Markedness": calculate_markedness(y_t, y_p)
        }

    # Helper 3: Bootstrap a specific subset of predictions to get the CI
    def get_metrics_with_ci(y_t, y_p):
        # 1. Get the actual point estimate
        base_metrics = get_core_metrics(y_t, y_p)
        
        # 2. Bootstrap to get the distribution
        n = len(y_t)
        boot_results = {key: [] for key in base_metrics.keys()}
        
        for _ in range(n_bootstrap):
            # Sample indices with replacement
            idx = np.random.choice(n, size=n, replace=True)
            sample_y_t = y_t[idx]
            sample_y_p = y_p[idx]
            
            # Calculate metrics for this bootstrap sample
            sample_metrics = get_core_metrics(sample_y_t, sample_y_p)
            for k, v in sample_metrics.items():
                boot_results[k].append(v)
        
        # 3. Calculate percentiles and format output
        formatted_results = {}
        lower_perc = (100 - ci) / 2
        upper_perc = 100 - lower_perc
        
        for k, v in base_metrics.items():
            low = np.percentile(boot_results[k], lower_perc)
            high = np.percentile(boot_results[k], upper_perc)
            # Format as: "0.85 (0.83 - 0.87)"
            formatted_results[k] = f"{v:.4f} ({low:.4f}-{high:.4f})"
            
        return formatted_results

    # --- Train/Test split ---
    split = int(len(y_real) * 0.8)
    y_real_train, y_real_test = y_real[:split], y_real[split:]
    y_pred_train, y_pred_test = y_pred[:split], y_pred[split:]

    cols = ["Model", "Set", "Accuracy", "Precision", "Recall", "F1", "MCC", "Class-Wise Error", "Markedness"]

    # Generate main sets
    df_main = pd.DataFrame([
        [model_name, "All", *get_metrics_with_ci(y_real, y_pred).values()],
        [model_name, "Train", *get_metrics_with_ci(y_real_train, y_pred_train).values()],
        [model_name, "Test", *get_metrics_with_ci(y_real_test, y_pred_test).values()],
    ], columns=cols)

    # Generate class-specific sets
    class_rows = []
    for cls in classes:
        # We bootstrap the subset where the TRUE label is 'cls'
        idx = (y_real == cls)
        if np.sum(idx) > 0:
            # For class-specific rows, MCC and Markedness are typically evaluated differently or omitted. 
            # We'll use the subset to calculate accuracy (which acts as recall for that class)
            y_r_cls = y_real[idx]
            y_p_cls = y_pred[idx]
            
            # Note: Macro metrics on a single class slice behave differently. 
            # To strictly mimic your table, we calculate the CI for this slice.
            res = get_metrics_with_ci(y_r_cls, y_p_cls)
            
            class_rows.append([
                model_name, 
                f"Class {cls}", 
                res["Accuracy"], 
                res["Precision"], 
                res["Recall"], 
                res["F1"], 
                "", # MCC is usually omitted for single classes in your table
                res["Class-Wise Error"], 
                res["Markedness"]
            ])
        else:
             class_rows.append([model_name, f"Class {cls}"] + ["N/A"] * 7)

    df_class = pd.DataFrame(class_rows, columns=cols)

    # Combine everything
    df_combined = pd.concat([df_main, df_class], ignore_index=True)
    return df_combined

# ====================================================
# Main Execution
# ====================================================

# 1. Load the data
# Make sure this path is correct for your machine
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\Data.xlsx", header=0, sheet_name="predicts")
columns = df.columns.tolist()

all_reports = []

# 2. Iterate dynamically over the columns in pairs (y_real, y_predict)
for i in range(0, len(columns), 2):
    raw_name = columns[i].strip()
    model_name = raw_name.split("_")[0] if "_" in raw_name else raw_name
    
    y_real = np.array(df.iloc[:, i].dropna())
    y_predict = np.array(df.iloc[:, i + 1].dropna())
    
    print(f"Bootstrapping metrics & CI for: {model_name} (This may take a few seconds)...")
    
    # 3. Build report
    # We use 1000 bootstraps as per your original code. You can lower this to 100 if it's too slow.
    df_report = build_ci_classification_reports(y_real, y_predict, model_name=model_name, n_bootstrap=1000)
    all_reports.append(df_report)

# 4. Concatenate all models into one big table
final_metrics_df = pd.concat(all_reports, ignore_index=True)

# 5. Display and copy to clipboard
print("\n--- Final Consolidated Metrics with 95% CI ---")
print(final_metrics_df.head(10)) 

final_metrics_df.to_clipboard(index=False)
print("\n[!] All metrics (Mean + 95% CI) have been successfully copied to your clipboard!")
