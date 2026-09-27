import pandas as pd
import numpy as np
import os
import win32com.client
import warnings
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ================== Execution Controls ==================
SAVE_TO_EXCEL = True
FILE_PATH = r"task\Data.xlsx"

# ================== Excel Helpers ==================
def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

def open_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[*] Opened Excel file:", filepath)
    except Exception as e:
        print("Note: Could not auto-open Excel GUI:", e)


def compute_shap(model, X_train, X_test, model_name):
    """
    Computes SHAP feature importance for the test set.
    """
    try:
        if model_name == "LGBC":
            explainer = shap.TreeExplainer(model)
            # Use max 100 samples to keep it fast
            X_sample = X_test if X_test.shape[0] <= 100 else X_test[np.random.choice(X_test.shape[0], 100, replace=False)]
            shap_values = explainer(X_sample, check_additivity=False)
        else:
            # BaggingClassifier does not support TreeExplainer out of the box
            # We use an exact or permutation explainer with a small background
            background = shap.kmeans(X_train, 5)  # reduced from 10
            explainer = shap.KernelExplainer(model.predict_proba, background)
            # Use max 30 samples and limit nsamples to keep it extremely fast
            X_sample = X_test if X_test.shape[0] <= 30 else X_test[np.random.choice(X_test.shape[0], 30, replace=False)]
            shap_values = explainer.shap_values(X_sample, silent=True, nsamples=50)
            
        # Get absolute mean importance across samples
        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values
        else:
            shap_array = np.array(shap_values)
            
        if len(shap_array.shape) == 3:
            # Multiclass
            shap_importance = np.abs(shap_array).mean(axis=(0, 2))
        elif len(shap_array.shape) == 2 and isinstance(shap_values, list):
            # Binary classification KernelExplainer output
            shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            # Binary tree explainer
            shap_importance = np.abs(shap_array).mean(axis=0)
            
        return shap_importance
    except Exception as e:
        print(f"Warning: SHAP computation failed for {model_name} due to {e}. Returning zeros.")
        return np.zeros(X_test.shape[1])

def main():
    close_excel_file(FILE_PATH)

    xl = pd.ExcelFile(FILE_PATH)
    if "Encoded_Data" in xl.sheet_names:
        sheet_name = "Encoded_Data"
    else:
        sheet_name = "Data"

    print(f"Reading dataset for K-Fold from sheet: '{sheet_name}'")
    df = pd.read_excel(FILE_PATH, sheet_name=sheet_name)
    df = df.dropna()
    
    target_column = df.columns[-1]
    X_raw = df.drop(columns=[target_column]).values
    y = df[target_column].values
    feature_names = df.drop(columns=[target_column]).columns.tolist()

    # Encode labels if they are not numeric
    if df[target_column].dtype == 'O' or df[target_column].dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_factories = {
        "LGBC": lambda f: LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42 + f,
            n_jobs=-1
        ),
        "BC": lambda f: BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=42 + f),
            n_estimators=50,
            max_samples=0.80,
            max_features=0.80,
            random_state=42 + f,
            n_jobs=-1
        ),
    }

    variants = {
        "No_SMOTE": False,
        "SMOTE_Fold": True
    }

    metrics_df_dict = {}
    df_reordered_dict = {}
    shap_results = []
    splits = list(skf.split(X_scaled, y))

    print("\nStarting K-Fold Cross Validation (Ablation Study)...")

    for variant_name, use_smote in variants.items():
        for model_name, factory in model_factories.items():
            run_tag = f"{model_name}_{variant_name}"
            print(f"\nEvaluating: {run_tag}")
            fold_records = []
            
            shap_importances_folds = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
                X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                if use_smote:
                    sm = SMOTE(random_state=42 + fold_idx)
                    try:
                        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                    except ValueError as e:
                        print(f"Fold {fold_idx}: SMOTE failed ({e}). Proceeding without SMOTE.")

                m = factory(fold_idx)
                m.fit(X_tr, y_tr)

                pred = m.predict(X_te)

                acc = float(accuracy_score(y_te, pred))
                prec = float(precision_score(y_te, pred, average='weighted', zero_division=0))
                rec = float(recall_score(y_te, pred, average='weighted', zero_division=0))
                f1 = float(f1_score(y_te, pred, average='weighted', zero_division=0))
                mcc = float(matthews_corrcoef(y_te, pred))

                fold_records.append({
                    "Fold": fold_idx,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1,
                    "MCC": mcc
                })
                
                # Compute SHAP
                print(f"  - Fold {fold_idx}: Computing SHAP...")
                importance = compute_shap(m, X_tr, X_te, model_name)
                shap_importances_folds.append(importance)

            df_m = pd.DataFrame(fold_records)
            metrics_df_dict[run_tag] = df_m

            # Average SHAP for this variant/model over 5 folds
            avg_shap = np.mean(shap_importances_folds, axis=0)
            for f_idx, f_name in enumerate(feature_names):
                shap_results.append({
                    "Model": model_name,
                    "Variant": variant_name,
                    "Feature": f_name,
                    "Mean_Abs_SHAP": avg_shap[f_idx]
                })

            # Identify Best Fold based on Accuracy
            best_fold_idx = int(df_m.loc[df_m["Accuracy"].idxmax(), "Fold"]) - 1
            best_test_idx = splits[best_fold_idx][1]
            rem_idx = df.index.difference(best_test_idx)

            # Place Best Fold test set in the last 20% of rows (from Original df, unmodified by SMOTE)
            df_reordered = pd.concat([df.loc[rem_idx], df.loc[best_test_idx]], axis=0).reset_index(drop=True)
            df_reordered_dict[run_tag] = df_reordered
            print(f"[+] {run_tag}: Best Fold = Fold {best_fold_idx + 1} (Accuracy = {df_m.loc[best_fold_idx, 'Accuracy']:.6f})")

    # Overall Summary Table with Mean & Std
    summary_list = []
    for run_tag, m_df in metrics_df_dict.items():
        summary_list.append({
            "Run_Tag": run_tag,
            "Accuracy_Mean": m_df["Accuracy"].mean(),
            "Accuracy_Std": m_df["Accuracy"].std(),
            "Precision_Mean": m_df["Precision"].mean(),
            "Precision_Std": m_df["Precision"].std(),
            "Recall_Mean": m_df["Recall"].mean(),
            "Recall_Std": m_df["Recall"].std(),
            "F1_Mean": m_df["F1 Score"].mean(),
            "F1_Std": m_df["F1 Score"].std(),
            "MCC_Mean": m_df["MCC"].mean(),
            "MCC_Std": m_df["MCC"].std(),
        })
    summary_df = pd.DataFrame(summary_list)
    shap_df = pd.DataFrame(shap_results).sort_values(by=["Variant", "Model", "Mean_Abs_SHAP"], ascending=[True, True, False])

    print("\n================== CV ABLATION RESULTS SUMMARY ==================")
    print(summary_df.to_string(index=False))
    print("=================================================================")

    # Save to Excel
    if SAVE_TO_EXCEL:
        close_excel_file(FILE_PATH)
        with pd.ExcelWriter(FILE_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            for run_tag in metrics_df_dict.keys():
                # Format tag correctly for excel sheet length constraints
                short_tag = run_tag.replace("No_SMOTE", "NoSmote").replace("SMOTE_Fold", "SmoteFold")
                
                metrics_df_dict[run_tag].to_excel(writer, sheet_name=f"Metrics_{short_tag}", index=False)
                df_reordered_dict[run_tag].to_excel(writer, sheet_name=f"Data_KFold_{short_tag}", index=False)
            
            summary_df.to_excel(writer, sheet_name="Ablation_CV_Summary", index=False)
            shap_df.to_excel(writer, sheet_name="SHAP_Summary_Ablation", index=False)
        
        print("\n[+] All models processed and saved to Excel sheets (Ablation).")
        open_excel_file(FILE_PATH)

if __name__ == "__main__":
    main()
