import pandas as pd
import numpy as np
import os
import win32com.client

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

# --- Load original dataset ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)
df = pd.read_excel(excel_path, sheet_name="Z-Score")

# --- Improved Randomization ---
# 1. Use numpy permutation for better entropy
# 2. We remove random_state so it's different every time you run it
df_shuffled = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

# Optional: Double-shuffle for maximum chaos
df_shuffled = df_shuffled.sample(frac=1).reset_index(drop=True)

# --- Save to same Excel file ---
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    df_shuffled.to_excel(writer, sheet_name="DATA_Shuffled", index=False)

open_excel_file(excel_path)

print("✅ Dataset randomized with high entropy.")
print(f"📁 Shuffled data saved to 'DATA_Shuffled' in '{excel_path}'.")