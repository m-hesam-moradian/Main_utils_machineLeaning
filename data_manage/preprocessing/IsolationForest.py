import pandas as pd
from sklearn.ensemble import IsolationForest
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

# --- Load Excel file ---
filepath = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(filepath)
df = pd.read_excel(filepath, sheet_name="data_after_vif")

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column]).select_dtypes(include="number")
y = df[target_column]

# --- Fit Isolation Forest on features only ---
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X)
outlier_flags = clf.predict(X)  # -1 = outlier, 1 = normal

# --- Filter clean samples ---
X_clean = X[outlier_flags == 1]
y_clean = y[outlier_flags == 1]

# --- Recombine cleaned data ---
cleaned_df = X_clean.copy()
cleaned_df[target_column] = y_clean.values

# --- Save to Excel ---
with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    cleaned_df.to_excel(writer, sheet_name="Isolation_Forest", index=False)

# --- Copy to clipboard and open Excel ---
cleaned_df.to_clipboard(index=False)
open_excel_file(filepath)
print("✅ Outliers removed. Cleaned data saved and copied to clipboard.")