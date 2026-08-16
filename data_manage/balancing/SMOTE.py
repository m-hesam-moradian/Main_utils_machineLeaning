import pandas as pd
from imblearn.over_sampling import SMOTE
import os
import win32com.client

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

# --- Load data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

xl = pd.ExcelFile(excel_path)
sheet_name = "Encoded_Data" if "Encoded_Data" in xl.sheet_names else "Data"
print(f"Reading dataset for SMOTE from sheet: '{sheet_name}'")

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Apply SMOTE Oversampling ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# --- Combine, randomize rows, and save to Excel ---
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced[target_column] = y_resampled

# Randomize row order so synthetic samples are thoroughly mixed
df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="SMOTE_Data", index=False)

print(f"Balanced dataset (randomized) saved to sheet 'SMOTE_Data' in {excel_path}. Shape: {df_balanced.shape}")