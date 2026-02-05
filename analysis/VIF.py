import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os
import win32com.client

# --- Excel control --- (Keep your existing functions)
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

# --- Customized VIF calculation ---
def calculate_vif_horizontal(X, threshold=5.0):
    X = X.copy()
    num_rows = len(X)
    
    # --- STEP 0: REMOVE UNIQUE IDENTIFIERS (The Overfitting Fix) ---
    leaky_cols = []
    for col in X.columns:
        # If every row has a unique value, it's a "leaky" fingerprint
        if X[col].nunique() == num_rows:
            leaky_cols.append(col)
    
    if leaky_cols:
        print(f"🚫 حذف خودکار ستون‌های شناسایی (Leakage): {leaky_cols}")
        # X.drop(columns=leaky_cols, inplace=True)

    vif_snapshots = []
    step = 1

    while True:
        if X.empty: break
        X_const = add_constant(X)
        vif = pd.DataFrame()
        vif["feature"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X_const.values, i + 1) for i in range(X.shape[1])]
        vif["Step"] = step
        vif_snapshots.append(vif.reset_index(drop=True))
        
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            drop_feature = vif.loc[vif["VIF"].idxmax(), "feature"]
            print(f"📌 حذف ویژگی '{drop_feature}' با VIF = {max_vif:.2f}")
            X.drop(columns=[drop_feature], inplace=True)
            step += 1
        else:
            break

    # --- Formatting for Excel Output ---
    max_rows = max(len(df) for df in vif_snapshots)
    for i in range(len(vif_snapshots)):
        rows_to_add = max_rows - len(vif_snapshots[i])
        if rows_to_add > 0:
            empty_rows = pd.DataFrame([["", "", ""]] * rows_to_add, columns=["feature", "VIF", "Step"])
            vif_snapshots[i] = pd.concat([vif_snapshots[i], empty_rows], ignore_index=True)

    spaced_snapshots = []
    for df_snap in vif_snapshots:
        spaced_snapshots.append(df_snap)
        spaced_snapshots.append(pd.DataFrame({"": [""] * max_rows}))

    final_vif = pd.concat(spaced_snapshots[:-1], axis=1)
    return X, final_vif

# --- Main Logic ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)
df = pd.read_excel(excel_path, sheet_name="Encoded_Data")

target_column = df.columns[-1]
X_input = df.drop(columns=[target_column])

selected_X, final_vif_horizontal = calculate_vif_horizontal(X_input, threshold=3)

# Reattach target
data_after_vif = selected_X.copy()
data_after_vif[target_column] = df[target_column]

# Save & Open
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    selected_X.to_excel(writer, sheet_name="selected_X", index=False)
    final_vif_horizontal.to_excel(writer, sheet_name="vif_horizontal", index=False)
    data_after_vif.to_excel(writer, sheet_name="data_after_vif", index=False)

data_after_vif.to_clipboard(index=False)
print("✅ Done! Dataset cleaned of IDs and Multicollinearity.")
open_excel_file(excel_path)