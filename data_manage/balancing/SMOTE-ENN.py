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

# Build balancing report
orig_dist = y.value_counts().sort_index()
final_dist = df_balanced[target_column].value_counts().sort_index()
all_classes = sorted(set(list(orig_dist.index) + list(final_dist.index)))

report_df = pd.DataFrame({
    "Encoded Value": all_classes,
    "Original Count": [orig_dist.get(c, 0) for c in all_classes],
    "After SMOTE-ENN Count": [final_dist.get(c, 0) for c in all_classes],
})
report_df["Change"] = report_df["After SMOTE-ENN Count"] - report_df["Original Count"]
report_df["Change %"] = ((report_df["Change"] / report_df["Original Count"]) * 100).round(1).astype(str) + "%"

summary_df = pd.DataFrame({
    "Metric": ["Method", "Original Samples", "After SMOTE-ENN", "LOF Outliers Removed",
                "Final Balanced Samples", "Number of Classes", "Shuffle Applied", "Random State"],
    "Value": ["SMOTE-ENN + LOF", str(len(X)), str(len(X_res)),
              str(int(np.sum(~inliers))), str(len(df_balanced)),
              str(len(all_classes)), "Yes", "42"]
})

# Save to Data.xlsx — one data sheet + one report sheet
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="SMOTE_Data", index=False)
    summary_df.to_excel(writer, sheet_name="Balancing_Report", index=False, startrow=0)
    report_df.to_excel(writer, sheet_name="Balancing_Report", index=False, startrow=len(summary_df) + 2)

print("\n[+] Balanced dataset saved to 'SMOTE_Data'. Summary saved to 'Balancing_Report' in Data.xlsx")