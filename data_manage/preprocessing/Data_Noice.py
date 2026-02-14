import pandas as pd
import numpy as np
import os
import win32com.client

def close_excel_file(filepath):
    """Saves and closes the specific Excel file if it is currently open."""
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("💾 Saved and 🔒 Closed Excel file:", filepath)
                break
    except Exception:
        pass

def open_excel_file(filepath):
    """Opens the specific Excel file and makes it visible."""
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = True
    excel.Workbooks.Open(os.path.abspath(filepath))
    print("📂 Opened Excel file:", filepath)

# --- Configuration ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
source_sheet = "DATA_Normalized"
output_sheet = "DATA_Noisy_Parallel"

# 1. Close file
close_excel_file(excel_path)

# 2. Load original dataset
df = pd.read_excel(excel_path, sheet_name=source_sheet)

# 3. Identify Target and Features
target_col_name = df.columns[-1]
x_features = df.drop(columns=[target_col_name]).select_dtypes(include=[np.number]).columns.tolist()

# --- 🧪 Create NEW Noisy Columns ---
# upward_bias: The intensity of the shift toward 1.0
upward_bias = 0.50 

for col in x_features:
    # Generate the noise
    chaos_map = np.random.uniform(0, upward_bias, size=len(df))
    
    # Create a NEW column name so the original column stays untouched
    noisy_col_name = f"{col}_noisy"
    
    # Calculate noise based on the original column and clip it
    df[noisy_col_name] = (df[col] + chaos_map).clip(0, 1)

# 4. Final Organization
# We keep [Original Features] + [New Noisy Features] + [Target]
# No shuffling, no target changes.

# --- Save to Excel ---
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    df.to_excel(writer, sheet_name=output_sheet, index=False)

# 5. Re-open the file
open_excel_file(excel_path)

print(f"✅ Parallel Noise Complete.")
print(f"➕ New noisy columns created with suffix '_noisy'.")
print(f"🔒 Original columns and Target '{target_col_name}' are untouched.")