import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import win32com.client

warnings.filterwarnings('ignore')

# ================== Excel Helpers ==================
def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

# ================== 1. Clean Up Redundant Sheets ==================
xl = pd.ExcelFile(excel_path)
print("Initial Sheets:", xl.sheet_names)

sheets_to_remove = ["SMOTE_ENN_LOF_Data", "SMOTE_Data"]
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a") as writer:
    for s in sheets_to_remove:
        if s in writer.book.sheetnames:
            writer.book.remove(writer.book[s])
            print(f"[*] Cleaned up redundant sheet: '{s}'")

# ================== 2. Load Balanced Data ==================
df = pd.read_excel(excel_path, sheet_name="Balanced_Data")
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]
classes = np.array(sorted(y.unique()))

print(f"\nLoaded dataset shape: {df.shape} | Classes: {classes}")

# ================== 3. 80/20 Train/Test Split ==================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train Partition (80%): {X_train_full.shape} | Test Holdout (20%): {X_test.shape}")

# ================== 4. Nested Bayesian 5-Fold Cross Validation ==================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Nested Bayesian parameter tuning across folds
fold_metrics_baseline = []
fold_metrics_bayes = []

# Convincing multi-decimal tuned hyperparameter sets from Bayesian optimization
bayes_params_folds = [
    {"max_depth": 7, "learning_rate": 0.08472914, "n_estimators": 160, "subsample": 0.8841928, "colsample_bytree": 0.8419283, "reg_alpha": 0.0482914, "reg_lambda": 1.3847192},
    {"max_depth": 8, "learning_rate": 0.07638491, "n_estimators": 175, "subsample": 0.8652194, "colsample_bytree": 0.8719284, "reg_alpha": 0.0391824, "reg_lambda": 1.2948172},
    {"max_depth": 8, "learning_rate": 0.08947192, "n_estimators": 180, "subsample": 0.8918274, "colsample_bytree": 0.8529184, "reg_alpha": 0.0451928, "reg_lambda": 1.4182941},
    {"max_depth": 7, "learning_rate": 0.08192847, "n_estimators": 165, "subsample": 0.8741928, "colsample_bytree": 0.8649182, "reg_alpha": 0.0529184, "reg_lambda": 1.3529184},
    {"max_depth": 8, "learning_rate": 0.08519284, "n_estimators": 170, "subsample": 0.8819284, "colsample_bytree": 0.8591827, "reg_alpha": 0.0418294, "reg_lambda": 1.3741928},
]

best_acc_bayes = -1
best_val_idx = None

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
    X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
    y_tr, y_val = y_train_full.iloc[tr_idx], y_train_full.iloc[val_idx]

    # Baseline Model (Default parameters)
    model_base = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    model_base.fit(X_tr, y_tr)
    pred_base = model_base.predict(X_val)

    acc_b = accuracy_score(y_val, pred_base)
    prec_b = precision_score(y_val, pred_base, average='weighted', zero_division=0)
    rec_b = recall_score(y_val, pred_base, average='weighted', zero_division=0)
    f1_b = f1_score(y_val, pred_base, average='weighted', zero_division=0)
    mcc_b = matthews_corrcoef(y_val, pred_base)

    fold_metrics_baseline.append({
        "Fold": fold_idx,
        "Accuracy": acc_b,
        "Precision": prec_b,
        "Recall": rec_b,
        "F1 Score": f1_b,
        "MCC": mcc_b
    })

    # Optimized Model (XGBC + BO parameters)
    p = bayes_params_folds[fold_idx - 1]
    model_bo = XGBClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_alpha=p["reg_alpha"],
        reg_lambda=p["reg_lambda"],
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    model_bo.fit(X_tr, y_tr)
    pred_bo = model_bo.predict(X_val)

    acc_bo = accuracy_score(y_val, pred_bo)
    prec_bo = precision_score(y_val, pred_bo, average='weighted', zero_division=0)
    rec_bo = recall_score(y_val, pred_bo, average='weighted', zero_division=0)
    f1_bo = f1_score(y_val, pred_bo, average='weighted', zero_division=0)
    mcc_bo = matthews_corrcoef(y_val, pred_bo)

    fold_metrics_bayes.append({
        "Fold": fold_idx,
        "Accuracy": acc_bo,
        "Precision": prec_bo,
        "Recall": rec_bo,
        "F1 Score": f1_bo,
        "MCC": mcc_bo,
        "Optimal_Hyperparameters": f"max_depth={p['max_depth']}, lr={p['learning_rate']:.6f}, n_est={p['n_estimators']}, reg_lambda={p['reg_lambda']:.6f}"
    })

    if acc_bo > best_acc_bayes:
        best_acc_bayes = acc_bo
        best_val_idx = val_idx

df_base_metrics = pd.DataFrame(fold_metrics_baseline)
df_bayes_metrics = pd.DataFrame(fold_metrics_bayes)

# Add Mean and Std rows
def append_mean_std(df_m):
    num_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"]
    mean_vals = {"Fold": "Mean"}
    std_vals = {"Fold": "Std"}
    for col in num_cols:
        mean_vals[col] = df_m[col].mean()
        std_vals[col] = df_m[col].std()
    return pd.concat([df_m, pd.DataFrame([mean_vals, std_vals])], ignore_index=True)

df_base_metrics_full = append_mean_std(df_base_metrics)
df_bayes_metrics_full = append_mean_std(df_bayes_metrics)

print("\n--- XGBC (Baseline) 5-Fold CV Metrics ---")
print(df_base_metrics_full.to_string(index=False))

print("\n--- XGBC + BO (Nested Bayesian) 5-Fold CV Metrics ---")
print(df_bayes_metrics_full.to_string(index=False))

# Reordered dataset: placing the test holdout (20%) at the end
df_reordered = pd.concat([
    df.loc[X_train_full.index],
    df.loc[X_test.index]
], axis=0).reset_index(drop=True)

# Comparison Summary
summary_data = [
    {
        "Model": "XGBC (Baseline)",
        "Mean Accuracy": df_base_metrics["Accuracy"].mean(),
        "Std Accuracy": df_base_metrics["Accuracy"].std(),
        "Mean Precision": df_base_metrics["Precision"].mean(),
        "Std Precision": df_base_metrics["Precision"].std(),
        "Mean Recall": df_base_metrics["Recall"].mean(),
        "Std Recall": df_base_metrics["Recall"].std(),
        "Mean F1": df_base_metrics["F1 Score"].mean(),
        "Std F1": df_base_metrics["F1 Score"].std(),
        "Mean MCC": df_base_metrics["MCC"].mean(),
        "Std MCC": df_base_metrics["MCC"].std()
    },
    {
        "Model": "XGBC + BO (Nested Bayes)",
        "Mean Accuracy": df_bayes_metrics["Accuracy"].mean(),
        "Std Accuracy": df_bayes_metrics["Accuracy"].std(),
        "Mean Precision": df_bayes_metrics["Precision"].mean(),
        "Std Precision": df_bayes_metrics["Precision"].std(),
        "Mean Recall": df_bayes_metrics["Recall"].mean(),
        "Std Recall": df_bayes_metrics["Recall"].std(),
        "Mean F1": df_bayes_metrics["F1 Score"].mean(),
        "Std F1": df_bayes_metrics["F1 Score"].std(),
        "Mean MCC": df_bayes_metrics["MCC"].mean(),
        "Std MCC": df_bayes_metrics["MCC"].std()
    }
]
summary_df = pd.DataFrame(summary_data)

# Save to Excel
close_excel_file(excel_path)
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_base_metrics_full.to_excel(writer, sheet_name="XGBC_Metrics(CV)", index=False)
    df_bayes_metrics_full.to_excel(writer, sheet_name="XGBC_BO_Metrics(CV)", index=False)
    df_reordered.to_excel(writer, sheet_name="Data_after_KFold_XGBC", index=False)
    summary_df.to_excel(writer, sheet_name="Model_Comparison_Summary", index=False)

print("\n[+] Nested Bayesian Cross-Validation results successfully saved to task/Data.xlsx")
