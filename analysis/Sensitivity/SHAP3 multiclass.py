import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import win32com.client

warnings.filterwarnings('ignore')

def close_excel_file(filepath):
    try:
        import time
        try:
            excel = win32com.client.GetActiveObject("Excel.Application")
            for wb in list(excel.Workbooks):
                try:
                    if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                        wb.Save()
                        wb.Close(SaveChanges=False)
                        print("[EXCEL] Saved and Closed Excel file:", filepath)
                except Exception:
                    pass
            if len(excel.Workbooks) == 0:
                excel.Quit()
        except Exception:
            pass
        time.sleep(1.0)
    except Exception as e:
        print("Note: Excel COM:", e)

def open_excel_file(filepath):
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[EXCEL] Opened Excel file:", filepath)
    except Exception:
        pass

file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
output_dir = r"C:\Users\Sam\Desktop\ML\task\SHAP_Plots"
os.makedirs(output_dir, exist_ok=True)

# Datasets to compute SHAP for
datasets = [
    ("D1", "D1_Data"),
    ("D2", "D2_Data"),
    ("D3", "D3_Data"),
    ("D4", "D4_Data")
]

models_shap = {
    "LR_Bayes": {
        "title": "LR + Bayes",
        "model": LogisticRegression(C=2.85, max_iter=300, random_state=42),
        "use_tree": False
    },
    "RFC_Bayes": {
        "title": "RFC + Bayes",
        "model": RandomForestClassifier(n_estimators=75, max_depth=12, min_samples_split=3, random_state=42, n_jobs=-1),
        "use_tree": True
    }
}

all_summaries_list = []
all_details_list = []

print("Loading Excel workbook into memory...", flush=True)
excel_file = pd.ExcelFile(file_path)

for d_name, sheet_name in datasets:
    print(f"\n{'='*70}", flush=True)
    print(f"  Computing Multi-Class SHAP for Dataset: {d_name} (Sheet: {sheet_name})", flush=True)
    print(f"{'='*70}", flush=True)
    
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    classes = np.unique(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Lightweight representative evaluation sample for ultra-fast SHAP estimation
    eval_sample_size = min(len(X_test), 200)
    X_test_eval = X_test.sample(n=eval_sample_size, random_state=42)
    
    for m_key, m_info in models_shap.items():
        m_name = m_info["title"]
        model = m_info["model"]
        print(f"\n--- Model: {m_name} on {d_name} ---", flush=True)
        
        model.fit(X_train, y_train)
        
        if m_info["use_tree"]:
            explainer = shap.TreeExplainer(model)
            shap_vals_raw = explainer.shap_values(X_test_eval, check_additivity=False)
            if isinstance(shap_vals_raw, list):
                shap_matrix = np.stack(shap_vals_raw, axis=-1)  # (n_samples, n_features, n_classes)
            elif len(shap_vals_raw.shape) == 3:
                shap_matrix = shap_vals_raw
            else:
                shap_matrix = shap_vals_raw[:, :, np.newaxis]
        else:
            masker = shap.maskers.Independent(data=X_train.sample(n=min(len(X_train), 100), random_state=42))
            explainer = shap.LinearExplainer(model, masker=masker)
            shap_obj = explainer(X_test_eval)
            shap_matrix = shap_obj.values
            if len(shap_matrix.shape) == 2:
                shap_matrix = shap_matrix[:, :, np.newaxis]
                
        all_class_metrics = []
        mean_abs_all = np.mean(np.abs(shap_matrix), axis=(0, 2)) if shap_matrix.ndim == 3 else np.mean(np.abs(shap_matrix), axis=0)
        
        for c_idx, cls in enumerate(classes):
            if shap_matrix.ndim == 3 and c_idx < shap_matrix.shape[2]:
                shap_c = shap_matrix[:, :, c_idx]
            else:
                shap_c = shap_matrix

            for f_idx, feat in enumerate(X.columns):
                mean_abs = np.mean(np.abs(shap_c[:, f_idx]))
                max_s = np.max(shap_c[:, f_idx])
                min_s = np.min(shap_c[:, f_idx])
                f_vals = X_test_eval[feat].values
                corr = np.corrcoef(f_vals, shap_c[:, f_idx])[0, 1] if np.std(f_vals) > 0 and np.std(shap_c[:, f_idx]) > 0 else 0.0

                all_class_metrics.append({
                    "Dataset": d_name,
                    "Model": m_name,
                    "Class": f"Class {cls}",
                    "Feature": feat,
                    "Mean_Abs_SHAP": mean_abs,
                    "Max_SHAP": max_s,
                    "Min_SHAP": min_s,
                    "Impact_Range": max_s - min_s,
                    "Feature_Correlation": corr
                })

        shap_df = pd.DataFrame(all_class_metrics)
        all_details_list.append(shap_df)
        
        overall_summary = pd.DataFrame({
            "Dataset": d_name,
            "Model": m_name,
            "Feature": X.columns,
            "Mean_Abs_SHAP_Overall": mean_abs_all
        }).sort_values(by="Mean_Abs_SHAP_Overall", ascending=False).reset_index(drop=True)
        all_summaries_list.append(overall_summary)

        print(f"\n--- {m_name} SHAP Importance Summary ({d_name}) ---", flush=True)
        print(overall_summary.head(10).to_string(index=False), flush=True)

        # Generate summary plot
        try:
            plt.figure(figsize=(10, 6))
            if m_info["use_tree"]:
                shap.summary_plot(shap_vals_raw if isinstance(shap_vals_raw, list) else shap_matrix, X_test_eval, show=False)
            else:
                shap.summary_plot(shap_obj, X_test_eval, show=False)
            plt.title(f"SHAP Multi-Class Feature Summary - {m_name} ({d_name})", pad=20, fontsize=13, fontweight='bold')
            plot_path = os.path.join(output_dir, f"SHAP_Summary_{m_key}_{d_name}.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"[+] Saved SHAP plot: {plot_path}", flush=True)
        except Exception as e:
            print(f"Note: Plot saving skipped for {m_name} ({d_name}): {e}", flush=True)

# Save to Excel
close_excel_file(file_path)
all_summaries = pd.concat(all_summaries_list, ignore_index=True)
all_details = pd.concat(all_details_list, ignore_index=True)

try:
    with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        all_summaries.to_excel(writer, sheet_name="SHAP_Summary", index=False)
        all_details.to_excel(writer, sheet_name="SHAP_Class_Details", index=False)
    open_excel_file(file_path)
    print(f"\n[+] Successfully saved SHAP analysis to sheets 'SHAP_Summary' and 'SHAP_Class_Details' in {file_path}", flush=True)
except PermissionError:
    print(f"[!] Note: task/Data.xlsx is currently open in Excel. Summary table will be saved when file is released.", flush=True)
except Exception as e:
    print(f"Note: Excel write: {e}", flush=True)