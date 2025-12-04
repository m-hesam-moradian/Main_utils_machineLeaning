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



# --- Load original dataset ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)
df = pd.read_excel(excel_path, sheet_name="SMOTE")

# --- Shuffle the dataset ---
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_clipboard(index=False)
# --- Save to same Excel file under new sheet ---
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    df_shuffled.to_excel(writer, sheet_name="DATA_Shuffled", index=False)
open_excel_file(excel_path)
print("✅ Dataset randomized successfully.")
print(f"📁 Shuffled data saved to sheet 'DATA_Shuffled' in '{excel_path}'.")
