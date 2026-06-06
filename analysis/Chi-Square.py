import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import os
import win32com.client

# --- Excel control ---
def close_excel_file(filepath):
    excel = win32com.client.Dispatch("Excel.Application")
    for wb in excel.Workbooks:
        try:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("💾 Saved and 🔒 Closed Excel file:", filepath)
                break
        except Exception:
            pass
    excel.Quit()

def open_excel_file(filepath):
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = True
    excel.Workbooks.Open(os.path.abspath(filepath))
    print("📂 Opened Excel file:", filepath)

# --- Chi-Square Feature Selection ---
def chi_square_selection(X, y, k=10):
    X = X.copy()

    # --- STEP 0: Remove leakage (same as your VIF logic) ---
    num_rows = len(X)
    leaky_cols = [col for col in X.columns if X[col].nunique() == num_rows]

    if leaky_cols:
        print(f"🚫 حذف خودکار ستون‌های شناسایی (Leakage): {leaky_cols}")
        # X.drop(columns=leaky_cols, inplace=True)

    # --- STEP 1: Scale to non-negative (IMPORTANT for chi2) ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # --- STEP 2: Apply Chi-Square ---
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    scores = selector.scores_
    pvalues = selector.pvalues_

    # --- Create report ---
    report = pd.DataFrame({
        "Feature": X.columns,
        "Chi2 Score": scores,
        "p-value": pvalues
    }).sort_values(by="Chi2 Score", ascending=False).reset_index(drop=True)

    # --- Selected features ---
    selected_features = X.columns[selector.get_support()]
    print(f"✅ Selected Features ({len(selected_features)}): {list(selected_features)}")

    X_selected_df = X[selected_features]

    return X_selected_df, report

# --- Main Logic ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

df = pd.read_excel(excel_path, sheet_name="Data")

target_column = df.columns[-1]
X_input = df.drop(columns=[target_column])
y = df[target_column]

# --- Apply Chi-Square ---
selected_X, chi_report = chi_square_selection(X_input, y, k=10)

# --- Reattach target ---
data_after_chi2 = selected_X.copy()
data_after_chi2[target_column] = y

# --- Save to Excel (same structure as VIF) ---
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    # selected_X.to_excel(writer, sheet_name="selected_X_chi2", index=False)
    chi_report.to_excel(writer, sheet_name="chi2_report", index=False)
    data_after_chi2.to_excel(writer, sheet_name="data_after_chi2", index=False)

# --- Clipboard ---
data_after_chi2.to_clipboard(index=False)

print("✅ Done! Chi-Square Feature Selection Completed.")
open_excel_file(excel_path)