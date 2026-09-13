import pandas as pd
import os
import win32com.client

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

def open_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[*] Opened Excel file:", filepath)
    except Exception:
        pass

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

xl = pd.ExcelFile(excel_path)
all_sheets = xl.sheet_names

# Target exact active model sheets
target_models = [
    "RNN", "RNN + BO",
    "GBC", "GBC + BO",
    "RFC", "RFC + BO",
    "QR", "QR + BO",
    "KNNC", "KNNC + BO",
    "ELM", "ELM + BO"
]

sheet_names = [s for s in target_models if s in all_sheets]

print("Matching model sheets for DataCatcher predictions:")
print(sheet_names)

merged_columns = []

for sheet in sheet_names:
    try:
        df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None, nrows=5)
    except Exception:
        continue

    header_row_idx = None
    for r_idx in range(min(5, len(df_raw))):
        row_vals = [str(v).lower() for v in df_raw.iloc[r_idx].values]
        if any("y_real" in v or "y_pred" in v for v in row_vals):
            header_row_idx = r_idx
            break

    if header_row_idx is None:
        continue

    df = pd.read_excel(excel_path, sheet_name=sheet, header=header_row_idx)

    cols = []
    for col in df.columns:
        col_name = str(col).lower()
        if "y_real" in col_name or "y_pred" in col_name:
            cols.append(col)
            if len(cols) == 2:
                break

    if len(cols) < 2:
        continue

    df_sub = df[cols].dropna(how="all").reset_index(drop=True)
    df_sub.columns = [f"{sheet}", f"{sheet}"]
    merged_columns.append(df_sub)
    print(f"Loaded predictions from sheet: {sheet}")

df_merged = pd.concat(merged_columns, axis=1)

print("\nPredictions Matrix Preview:")
print(df_merged.head())
print("Shape:", df_merged.shape)

close_excel_file(excel_path)
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_merged.to_excel(writer, sheet_name="predicts", index=False)

print(f"\n[+] Saved combined model predictions to sheet 'predicts' in {excel_path}")
open_excel_file(excel_path)