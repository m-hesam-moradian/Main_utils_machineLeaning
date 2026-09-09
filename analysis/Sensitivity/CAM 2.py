import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from SALib.sample import saltelli
from SALib.analyze import sobol
import warnings
import os
import win32com.client

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            try:
                if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                    wb.Save()
                    wb.Close(SaveChanges=False)
                    print("[+] Saved and Closed Excel file:", filepath)
                    break
            except Exception:
                pass
    except Exception as e:
        print("Note: Excel is not running or COM skipped:", e)

# ==========================================
# 1. Load Data
# ==========================================
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

df = pd.read_excel(excel_path, sheet_name="data_after_vif")
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
feature_names = list(X.columns)

y = pd.read_csv(r"C:\Users\Sam\Desktop\ML\data\predictions.txt", header=None).squeeze()

# Align lengths
min_len = min(X.shape[0], len(y))
X = X.iloc[:min_len]
y = y.iloc[:min_len]

# ==========================================
# 2. Fit Surrogate Model
# ==========================================
model = CatBoostRegressor(iterations=130, depth=3, l2_leaf_reg=12.5, learning_rate=0.04, verbose=0, random_state=42)
model.fit(X, y)

# ==========================================
# 3. Setup SALib Problem & Sample
# ==========================================
bounds = [[float(X[col].min()), float(X[col].max())] for col in feature_names]
problem = {
    'num_vars': len(feature_names),
    'names': feature_names,
    'bounds': bounds
}

# Generate Saltelli samples (required for S2 interactions)
print("[+] Generating Saltelli samples (N=1024)...")
param_values = saltelli.sample(problem, 1024, calc_second_order=True)
print(f"[+] Evaluating surrogate model across {len(param_values)} parameter combinations...")
Y_pred = model.predict(param_values)

# ==========================================
# 4. Calculate S1, S2, ST Indices
# ==========================================
print("[+] Computing Sobol sensitivity indices (S1, S2, ST)...")
Si = sobol.analyze(problem, Y_pred, calc_second_order=True)

# Format S1 and ST into a DataFrame
df_s1_st = pd.DataFrame({
    "Parameter": problem['names'],
    "S1": np.round(np.maximum(0, Si['S1']), 4),
    "ST": np.round(np.maximum(0, Si['ST']), 4)
}).sort_values(by="ST", ascending=False).reset_index(drop=True)

# Format S2 into a DataFrame (Extracting pairwise interactions)
s2_data = []
for i, name_i in enumerate(problem['names']):
    for j, name_j in enumerate(problem['names']):
        if i < j:
            val = Si['S2'][i, j]
            s2_data.append({
                "Parameter_1": name_i,
                "Parameter_2": name_j,
                "S2": round(float(np.maximum(0, val)) if not np.isnan(val) else 0.0, 4)
            })
df_s2 = pd.DataFrame(s2_data).sort_values(by="S2", ascending=False).reset_index(drop=True)

# ==========================================
# 5. Export and Print Results
# ==========================================
close_excel_file(excel_path)
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_s1_st.to_excel(writer, sheet_name="CAM_Sensitivity", index=False)
    df_s2.to_excel(writer, sheet_name="CAM_S2_Interactions", index=False)

print("\n--- First-Order (S1) and Total (ST) Indices ---")
print(df_s1_st.to_string(index=False))

print("\n--- Top 10 Second-Order (S2) Interactions ---")
print(df_s2.head(10).to_string(index=False))

print(f"\n[+] CAM Sensitivity results successfully saved to sheet 'CAM_Sensitivity' and 'CAM_S2_Interactions' in {excel_path}")