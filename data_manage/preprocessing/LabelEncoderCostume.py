import pandas as pd

def close_excel_file(filepath):
    import os
    import win32com.client
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
    import os
    import win32com.client
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = True
    excel.Workbooks.Open(os.path.abspath(filepath))
    print("📂 Opened Excel file:", filepath)

# --- Load your Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)
df = pd.read_excel(excel_path, sheet_name="Data")

# --- Create a copy to avoid modifying original ---
df_encoded = df.copy()

# --- Define explicit mapping for Fault Label ---
# You can choose any numeric codes, here we use:
# Fault = 0, Normal = 1, Warning = 2
class_mapping = {
    "Fault": 0,
    "Normal": 1,
    "Warning": 2
}

# --- Apply mapping only to the target column ---
target_column = "Fault Label"
df_encoded[target_column] = df_encoded[target_column].map(class_mapping)

# --- Save to a new sheet in the same Excel file ---
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_encoded.to_excel(writer, sheet_name="Encoded_Data", index=False)

open_excel_file(excel_path)
print("✅ Encoded 'Fault Label' column with custom mapping and saved to 'Encoded_Data'.")