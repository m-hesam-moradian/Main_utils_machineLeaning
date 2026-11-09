import os
import sys
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

excel_path = r"task/Data.xlsx"

def close_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception as e:
        pass

print("[1] Closing Excel if open...")
close_excel_file(excel_path)

print("[2] Reading input dataset 'data_after_vif' from", excel_path)
df = pd.read_excel(excel_path, sheet_name="data_after_vif")
target_column = "Adaptation Status"
X = df.drop(columns=[target_column])
y = df[target_column]

class_names = {0: "High_Shift", 1: "Low_Shift", 2: "Moderate_Shift"}

print("Original Training Partition Class Distribution:")
for cls_val in sorted(y.unique()):
    cnt = (y == cls_val).sum()
    print(f"  Class {cls_val} ({class_names[cls_val]}): {cnt} ({cnt/len(y)*100:.2f}%)")

# Step A: SMOTE-ENN
print("\n[3] Applying SMOTE-ENN hybrid resampling...")
smote_enn = SMOTEENN(
    smote=SMOTE(k_neighbors=5, random_state=42),
    enn=EditedNearestNeighbours(n_neighbors=3, kind_sel="mode"),
    random_state=42
)
X_res, y_res = smote_enn.fit_resample(X, y)
print(f"After SMOTE-ENN shape: {X_res.shape}")

# Step B: Local Outlier Factor (LOF) filtering to clean noisy boundary artifacts
print("[4] Applying LOF outlier filtering (contamination=0.02)...")
scaler = StandardScaler()
X_res_sc = scaler.fit_transform(X_res)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
outlier_mask = lof.fit_predict(X_res_sc)
inliers = (outlier_mask == 1)

X_clean = X_res[inliers].copy()
y_clean = y_res[inliers].copy()
outliers_removed = int(np.sum(~inliers))
print(f"LOF outliers removed: {outliers_removed} | Inliers retained: {len(X_clean)}")

df_balanced = pd.DataFrame(X_clean, columns=X.columns)
df_balanced[target_column] = y_clean.values

# Shuffle dataset rows thoroughly per Rule
print("[5] Shuffling balanced dataset rows (random_state=42)...")
df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

print("\nFinal Balanced Dataset Distribution:")
for cls_val in sorted(df_balanced[target_column].unique()):
    cnt = (df_balanced[target_column] == cls_val).sum()
    print(f"  Class {cls_val} ({class_names[cls_val]}): {cnt} ({cnt/len(df_balanced)*100:.2f}%)")

# Build Balancing Reports
orig_dist = y.value_counts().sort_index()
final_dist = df_balanced[target_column].value_counts().sort_index()
all_classes = sorted(set(list(orig_dist.index) + list(final_dist.index)))

report_df = pd.DataFrame({
    "Encoded Class": all_classes,
    "Class Name": [class_names.get(c, str(c)) for c in all_classes],
    "Original Count (Train_Data)": [orig_dist.get(c, 0) for c in all_classes],
    "Final Balanced Count": [final_dist.get(c, 0) for c in all_classes],
})
report_df["Change"] = report_df["Final Balanced Count"] - report_df["Original Count (Train_Data)"]
report_df["Change %"] = ((report_df["Change"] / report_df["Original Count (Train_Data)"]) * 100).round(1).astype(str) + "%"

summary_df = pd.DataFrame({
    "Metric": [
        "Resampling Pipeline",
        "Original Training Samples",
        "After SMOTE-ENN Resampling",
        "LOF Boundary Outliers Removed",
        "Final Balanced Samples",
        "Input Feature Count",
        "Number of Classes",
        "Row Shuffling Applied",
        "Random State"
    ],
    "Value": [
        "SMOTE-ENN + LOF Filtering",
        str(len(X)),
        str(len(X_res)),
        str(outliers_removed),
        str(len(df_balanced)),
        str(X.shape[1]),
        str(len(all_classes)),
        "Yes",
        "42"
    ]
})

print("\n[6] Writing 'SMOTE_Data' and 'Balancing_Report' to task/Data.xlsx...")
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="SMOTE_Data", index=False)
    summary_df.to_excel(writer, sheet_name="Balancing_Report", index=False, startrow=0)
    report_df.to_excel(writer, sheet_name="Balancing_Report", index=False, startrow=len(summary_df) + 2)

print("[+] Successfully saved 'SMOTE_Data' and 'Balancing_Report' to task/Data.xlsx!")
