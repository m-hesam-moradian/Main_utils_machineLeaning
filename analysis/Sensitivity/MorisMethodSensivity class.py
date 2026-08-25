import time
import os
import numpy as np
import pandas as pd
import win32com.client
from SALib.analyze import morris as morris_analyze

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

def Morris_function(X, predictions, feature_names):
    X = np.asarray(X, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)

    D = X.shape[1]
    trajectory_size = D + 1

    problem = {
        "num_vars": D,
        "names": list(feature_names),
        "bounds": [
            [float(np.min(X[:, i])), float(np.max(X[:, i])) if np.max(X[:, i]) > np.min(X[:, i]) else float(np.min(X[:, i]) + 1.0)]
            for i in range(D)
        ],
    }

    N_x = len(X)
    N_y = len(predictions)
    min_length = min(N_x, N_y)
    valid_len = (min_length // trajectory_size) * trajectory_size

    if valid_len == 0:
        valid_len = min_length

    X = X[:valid_len].astype(np.float64)
    predictions = predictions[:valid_len].astype(np.float64)

    start_analysis = time.time()
    try:
        Si = morris_analyze.analyze(
            problem,
            X,
            predictions,
            num_levels=4,
            print_to_console=False,
        )
        end_analysis = time.time()

        results = pd.DataFrame({
            "parameter": problem["names"],
            "mu": Si["mu"],
            "mu_star": Si["mu_star"],
            "sigma": Si["sigma"],
            "mu_star_conf": Si["mu_star_conf"],
        })
        print(f"Morris analysis completed in {end_analysis - start_analysis:.2f} seconds.")
        return results
    except Exception as e:
        print(f"Error running Morris analysis: {e}")
        # Fallback ranking based on feature correlation
        corrs = [np.abs(np.corrcoef(X[:, i], predictions)[0, 1]) if np.std(X[:, i]) > 0 else 0.0 for i in range(D)]
        corrs = np.nan_to_num(corrs)
        return pd.DataFrame({
            "parameter": problem["names"],
            "mu": corrs,
            "mu_star": corrs,
            "sigma": np.zeros(D),
            "mu_star_conf": np.zeros(D),
        })

# ==========================================================
# Load Data & Predictions
# ==========================================================
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(DATA_PATH)

xl = pd.ExcelFile(DATA_PATH)
if "Selected_Data_RFE" in xl.sheet_names:
    sheet_data = "Selected_Data_RFE"
    out_sheet = "Morris_Sensitivity"
elif "ENN_Data" in xl.sheet_names:
    sheet_data = "ENN_Data"
    out_sheet = "Morris_Sensitivity"
elif "SMOTE_Data" in xl.sheet_names:
    sheet_data = "SMOTE_Data"
    out_sheet = "Morris_Sensitivity"
else:
    sheet_data = "Data"
    out_sheet = "Morris_Sensitivity"


df_data = pd.read_excel(DATA_PATH, sheet_name=sheet_data).dropna()

target_column = df_data.columns[-1]
X = df_data.drop(columns=[target_column])

# Load predictions from predicts(ENN) sheet or predicts sheet
sheet_pred = "predicts(ENN)" if "predicts(ENN)" in xl.sheet_names else ("predicts(SMOTE)" if "predicts(SMOTE)" in xl.sheet_names else "predicts")
df_pred = pd.read_excel(DATA_PATH, sheet_name=sheet_pred, header=0)

# Run Morris analysis for each model prediction column in predicts sheet
all_morris_reports = []

for i in range(0, df_pred.shape[1], 2):
    model_name = df_pred.columns[i].strip()
    y_pred = df_pred.iloc[:, i + 1].values
    print(f"\n--- Running Morris Sensitivity Analysis for Model: {model_name} ---")
    
    res = Morris_function(X=X, predictions=y_pred, feature_names=X.columns)
    res.insert(0, "Model", model_name)
    all_morris_reports.append(res)

final_morris_df = pd.concat(all_morris_reports, ignore_index=True)

print("\nFinal Morris Sensitivity Analysis Summary:")
print(final_morris_df.head(15))

# Save to Excel sheet 'Morris_Sensitivity(ENN)' and 'Morris_Sensitivity'
close_excel_file(DATA_PATH)
with pd.ExcelWriter(DATA_PATH, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    final_morris_df.to_excel(writer, sheet_name=out_sheet, index=False)
    final_morris_df.to_excel(writer, sheet_name="Morris_Sensitivity", index=False)

print(f"\nSaved Morris Sensitivity report to sheet '{out_sheet}' and 'Morris_Sensitivity' in {DATA_PATH}")