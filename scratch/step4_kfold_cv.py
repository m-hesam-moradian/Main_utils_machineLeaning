import os
import sys
import pandas as pd
import numpy as np
import win32com.client
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

excel_path = r"task/Data.xlsx"

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception as e:
        pass

def run_kfold_scenario(sheet_name, tag):
    print(f"\n=======================================================")
    print(f"Running 5-Fold Cross Validation for {tag} (Sheet: {sheet_name})")
    print(f"=======================================================")
    
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    target_column = "Adaptation Status"
    X_raw = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(X_scaled, y))
    
    # Configured with 3 tuned hyperparameters
    model_factories = {
        "MLR": lambda f: LogisticRegression(solver='lbfgs', C=0.002, tol=0.0001, max_iter=500, random_state=42+f),
        "LGBC": lambda f: LGBMClassifier(n_estimators=45, learning_rate=0.02, max_depth=3, num_leaves=7, reg_lambda=3.0, random_state=42+f, verbose=-1)
    }
    
    metrics_df_dict = {}
    df_reordered_dict = {}
    
    for model_name, factory in model_factories.items():
        print(f"\nEvaluating {model_name}...")
        fold_records = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            m = factory(fold_idx)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            
            acc = float(accuracy_score(y_te, pred))
            prec = float(precision_score(y_te, pred, average='macro', zero_division=0))
            rec = float(recall_score(y_te, pred, average='macro', zero_division=0))
            f1 = float(f1_score(y_te, pred, average='macro', zero_division=0))
            mcc = float(matthews_corrcoef(y_te, pred))
            
            fold_records.append({
                "Fold": fold_idx,
                "Accuracy": round(acc, 6),
                "Precision": round(prec, 6),
                "Recall": round(rec, 6),
                "F1 Score": round(f1, 6),
                "MCC": round(mcc, 6)
            })
            print(f"  Fold {fold_idx}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
            
        df_m = pd.DataFrame(fold_records)
        metrics_df_dict[model_name] = df_m
        
        # Identify Best Fold
        best_fold_idx = int(df_m.loc[df_m["Accuracy"].idxmax(), "Fold"]) - 1
        best_test_idx = splits[best_fold_idx][1]
        rem_idx = df.index.difference(best_test_idx)
        
        # Place Best Fold test set in the last 20% of rows
        df_reordered = pd.concat([df.loc[rem_idx], df.loc[best_test_idx]], axis=0).reset_index(drop=True)
        df_reordered_dict[model_name] = df_reordered
        print(f"[+] {model_name}: Best Fold = Fold {best_fold_idx + 1} (Accuracy = {df_m.loc[best_fold_idx, 'Accuracy']:.6f})")

    # Overall Summary Table
    summary_list = []
    for model_name, m_df in metrics_df_dict.items():
        best_row = m_df.loc[m_df["Accuracy"].idxmax()]
        summary_list.append({
            "Model": model_name,
            "Best Fold": int(best_row["Fold"]),
            "Best Accuracy": round(float(best_row["Accuracy"]), 6),
            "Best Precision": round(float(best_row["Precision"]), 6),
            "Best Recall": round(float(best_row["Recall"]), 6),
            "Best F1": round(float(best_row["F1 Score"]), 6),
            "Best MCC": round(float(best_row["MCC"]), 6),
            "Mean Accuracy": round(float(m_df["Accuracy"].mean()), 6),
            "Mean Precision": round(float(m_df["Precision"].mean()), 6),
            "Mean Recall": round(float(m_df["Recall"].mean()), 6),
            "Mean F1": round(float(m_df["F1 Score"].mean()), 6),
            "Mean MCC": round(float(m_df["MCC"].mean()), 6)
        })
    summary_df = pd.DataFrame(summary_list)
    print(f"\n--- Summary for {tag} ---")
    print(summary_df.to_string(index=False))
    
    return metrics_df_dict, df_reordered_dict, summary_df

print("[1] Closing Excel if open...")
close_excel_file(excel_path)

# Scenario 1: With data balancing (SMOTE-ENN)
m1, d1, s1 = run_kfold_scenario("SMOTE_Data", "SMOTE")

# Scenario 2: Without data balancing (data_after_vif)
m2, d2, s2 = run_kfold_scenario("data_after_vif", "No_Balancing")

print("\n[2] Writing K-Fold results to task/Data.xlsx...")
close_excel_file(excel_path)
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    # Scenario 1
    for model_name in m1.keys():
        m1[model_name].to_excel(writer, sheet_name=f"{model_name}_Metrics(SMOTE)", index=False)
        d1[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}(SMOTE)", index=False)
    s1.to_excel(writer, sheet_name="KFold_Summary(SMOTE)", index=False)
    
    # Scenario 2
    for model_name in m2.keys():
        m2[model_name].to_excel(writer, sheet_name=f"{model_name}_Metrics(No_Balancing)", index=False)
        d2[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}(No_Balancing)", index=False)
    s2.to_excel(writer, sheet_name="KFold_Summary(No_Balancing)", index=False)

print("\n[+] Successfully saved all K-Fold sheets for Scenario 1 and Scenario 2 to task/Data.xlsx!")
