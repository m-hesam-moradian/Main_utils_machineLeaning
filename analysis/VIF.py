import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


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


excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)
df = pd.read_excel(excel_path, sheet_name="DATA_Shuffled")
target_column = df.columns[-1]
X = df.drop(columns=[target_column])






def calculate_vif(X, threshold=10.0, verbose=True):
    X = X.copy()
    while True:
        X_const = add_constant(X)
        vif = pd.DataFrame()
        vif["feature"] = X.columns
        vif["VIF"] = [
            variance_inflation_factor(X_const.values, i + 1) for i in range(X.shape[1])
        ]

        max_vif = vif["VIF"].max()
        if verbose:
            print(vif)
            print("=" * 40)

        if max_vif > threshold:
            drop_feature = vif.loc[vif["VIF"].idxmax(), "feature"]
            if verbose:
                print(f"📌 حذف ویژگی '{drop_feature}' با VIF = {max_vif:.2f}")
            X.drop(columns=[drop_feature], inplace=True)
        else:
            break

    return X, vif


selected_X, final_vif = calculate_vif(X, threshold=10.0)
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    selected_X.to_excel(writer, sheet_name="selected_X", index=False)

with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    final_vif.to_excel(writer, sheet_name="final_vif", index=False)

open_excel_file(excel_path)
print("✅ ویژگی‌های نهایی باقی‌مانده:")
print(selected_X.columns.tolist())
