import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import win32com.client

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

df = pd.read_excel(excel_path, sheet_name="Encoded_Data")
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

print("Original Class Distribution:")
print(y.value_counts())

# Step A: SMOTE-ENN
smote_enn = SMOTEENN(
    smote=SMOTE(k_neighbors=5, random_state=42),
    enn=EditedNearestNeighbours(n_neighbors=3, kind_sel="mode"),
    random_state=42
)
X_res, y_res = smote_enn.fit_resample(X, y)
print("\nAfter SMOTE-ENN Shape:", X_res.shape)
print("After SMOTE-ENN Class Distribution:\n", pd.Series(y_res).value_counts())

# Step B: Local Outlier Factor (LOF) filtering
scaler = StandardScaler()
X_res_sc = scaler.fit_transform(X_res)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
outlier_mask = lof.fit_predict(X_res_sc)
inliers = (outlier_mask == 1)

X_clean = X_res[inliers]
y_clean = y_res[inliers]

print(f"\nLOF Filtered Outliers: {np.sum(~inliers)} | Remaining Inliers: {len(X_clean)}")

df_balanced = pd.DataFrame(X_clean, columns=X.columns)
df_balanced[target_column] = y_clean

# Shuffle dataset
df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

print("\nFinal Balanced Dataset Shape:", df_balanced.shape)
print("Final Class Distribution:\n", df_balanced[target_column].value_counts())

# Save to Data.xlsx
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="Balanced_Data", index=False)
    df_balanced.to_excel(writer, sheet_name="SMOTE_ENN_LOF_Data", index=False)
    df_balanced.to_excel(writer, sheet_name="SMOTE_Data", index=False)

print("\n[+] Balanced dataset successfully saved to sheets 'Balanced_Data', 'SMOTE_ENN_LOF_Data', and 'SMOTE_Data' in Data.xlsx")