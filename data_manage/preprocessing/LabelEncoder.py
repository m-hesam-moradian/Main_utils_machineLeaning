import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# Load your Excel file
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)
df = pd.read_excel(excel_path ,sheet_name="Data")

# Create a copy to avoid modifying original
df_encoded = df.copy()
encoder = LabelEncoder()
# Loop through columns and encode if dtype is object or category
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object" or df_encoded[col].dtype.name == "category" or df_encoded[col].dtype.name == "bool":
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))

# Save to a new sheet in the same Excel file
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl") as writer:
    df_encoded.to_excel(writer, sheet_name="Encoded_Data", index=False)
 
open_excel_file(excel_path)