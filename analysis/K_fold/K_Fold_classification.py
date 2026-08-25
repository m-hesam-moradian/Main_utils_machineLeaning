import pandas as pd
import numpy as np
import os
import win32com.client
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ================== Execution Controls ==================
SAVE_TO_EXCEL = True  # Set to True to export results to task/Data.xlsx

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
    except Exception:
        pass

def main():
    # ================== Load Dataset ==================
    filepath = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
    xl = pd.ExcelFile(filepath)
    if "Selected_Data_RFE" in xl.sheet_names:
        sheet_name = "Selected_Data_RFE"
        suffix = "RFE"
    elif "ENN_Data" in xl.sheet_names:
        sheet_name = "ENN_Data"
        suffix = "ENN"
    elif "SMOTE_Data" in xl.sheet_names:
        sheet_name = "SMOTE_Data"
        suffix = "SMOTE"
    else:
        sheet_name = "Encoded_Data"
        suffix = "Encoded"

    print(f"Reading dataset for K-Fold from sheet: '{sheet_name}'")
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    target_column = df.columns[-1]
    X_full = df.drop(columns=[target_column])
    y = df[target_column]

    # ================== Target Models with 3 Hyperparameters ==================
    models = {
        "MLR": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=2000,
                random_state=42
            )
        ),
        "SVC": make_pipeline(
            StandardScaler(),
            SVC(
                C=5.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=42
            )
        )
    }

    # =============== K-Fold Execution ==================
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    metrics_df_dict = {}
    df_reordered_dict = {}

    for model_name, model in models.items():
        fold_metrics_list = []
        print(f"Training {model_name}...")

        best_acc = -1
        best_test_idx = None

        for fold_index, (train_idx, test_idx) in enumerate(kf.split(X_full), 1):
            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)


            fold_metrics_list.append({
                "Fold": fold_index,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })

            if acc > best_acc:
                best_acc = acc
                best_test_idx = test_idx

        metrics_df = pd.DataFrame(fold_metrics_list)
        metrics_df_dict[model_name] = metrics_df

        remaining_idx = df.index.difference(best_test_idx)
        df_reordered_dict[model_name] = pd.concat(
            [df.loc[remaining_idx], df.loc[best_test_idx]], axis=0
        )

    # ================== Create Summary ==================
    summary_list = []
    for model_name, m_df in metrics_df_dict.items():
        best_row = m_df.loc[m_df["Accuracy"].idxmax()]
        summary_list.append({
            "Model": model_name,
            "Best Fold": best_row["Fold"],
            "Best Accuracy": best_row["Accuracy"],
            "Best Precision": best_row["Precision"],
            "Best Recall": best_row["Recall"],
            "Best F1": best_row["F1 Score"],
            "Mean Accuracy": m_df["Accuracy"].mean(),
            "Mean Precision": m_df["Precision"].mean(),
            "Mean Recall": m_df["Recall"].mean(),
            "Mean F1": m_df["F1 Score"].mean()
        })
    summary_df = pd.DataFrame(summary_list)

    print("\n================== K-FOLD RESULTS SUMMARY ==================")
    print(summary_df.to_string(index=False))
    print("============================================================")

    # ================== Save to Excel ==================
    if SAVE_TO_EXCEL:
        close_excel_file(filepath)
        with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            for model_name in models:
                metrics_df_dict[model_name].to_excel(writer, sheet_name=f"{model_name}_Metrics({suffix})", index=False)
                df_reordered_dict[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}({suffix})", index=False)
            summary_df.to_excel(writer, sheet_name=f"Model_Comparison_Summary({suffix})", index=False)
        open_excel_file(filepath)
        print(f"\n[+] All models processed and saved to Excel with ({suffix}) sheets.")

if __name__ == "__main__":
    main()