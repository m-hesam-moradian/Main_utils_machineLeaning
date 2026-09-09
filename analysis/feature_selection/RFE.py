import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

# --- Excel helpers ---
def close_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.Dispatch("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

def open_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[*] Opened Excel file:", filepath)
    except Exception:
        pass

# =========================================================
# LOAD DATA (From Encoded_Data)
# =========================================================
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

xl = pd.ExcelFile(excel_path)
if "Selected_Data_RFE" in xl.sheet_names:
    pass # for downstream
if "ENN_Data" in xl.sheet_names:
    sheet_name = "ENN_Data"
elif "SMOTE_Data" in xl.sheet_names:
    sheet_name = "SMOTE_Data"
elif "Balanced_Data" in xl.sheet_names:
    sheet_name = "Balanced_Data"
elif "Encoded_Data" in xl.sheet_names:
    sheet_name = "Encoded_Data"
else:
    sheet_name = "Data"

print(f"Reading dataset for RFE from sheet: '{sheet_name}'")
df = pd.read_excel(excel_path, sheet_name=sheet_name)

target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

col_names = X.columns
X_np = np.array(X)
y_np = np.array(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

# =========================================================
# FEATURE SELECTION (RFE)
# =========================================================
estimator = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

selector = RFE(
    estimator=estimator,
    n_features_to_select=1,
    step=1
)

selector.fit(X_scaled, y_np)
feature_ranking = selector.ranking_

ranking_df = pd.DataFrame({
    "Feature": col_names,
    "Rank": feature_ranking
}).sort_values("Rank").reset_index(drop=True)

TOP_K = 18

selected_features = ranking_df[
    ranking_df["Rank"] <= TOP_K
]["Feature"].tolist()

removed_features = ranking_df[
    ranking_df["Rank"] > TOP_K
]["Feature"].tolist()

ranking_df["Status"] = ranking_df["Rank"].apply(
    lambda x: "Kept" if x <= TOP_K else "Removed"
)

# =========================================================
# EVALUATE SUBSETS (Progressive curve with 5-Fold CV)
# =========================================================
report_rows = []
sorted_feature_indices = np.argsort(selector.ranking_)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for input_count in range(1, X_scaled.shape[1] + 1):
    sub_indices = sorted_feature_indices[:input_count]
    X_subset = X_scaled[:, sub_indices]

    fold_accs, fold_f1s, fold_recs = [], [], []
    for tr, te in skf.split(X_subset, y_np):
        model = ExtraTreesClassifier(
            n_estimators=50,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_subset[tr], y_np[tr])
        pred = model.predict(X_subset[te])

        fold_accs.append(accuracy_score(y_np[te], pred))
        fold_f1s.append(f1_score(y_np[te], pred, average="macro", zero_division=0))
        fold_recs.append(recall_score(y_np[te], pred, average="macro", zero_division=0))

    report_rows.append({
        "Features used": input_count,
        "Accuracy": np.mean(fold_accs),
        "F1-Score": np.mean(fold_f1s),
        "Recall": np.mean(fold_recs)
    })

report_df = pd.DataFrame(report_rows)

# Combine report and rankings side-by-side
max_rows = max(len(report_df), len(ranking_df))
report_df_extended = report_df.reindex(range(max_rows))
ranking_df_extended = ranking_df.reindex(range(max_rows))
space1 = pd.DataFrame({"": [""] * max_rows})

combined_report = pd.concat(
    [report_df_extended, space1, ranking_df_extended],
    axis=1
)

print("\n==============================")
print(f"[+] KEPT FEATURES (Top {TOP_K})")
print("==============================")
for f in selected_features:
    print(f" - {f}")

print("\n==============================")
print(f"[-] REMOVED FEATURES ({len(removed_features)})")
print("==============================")
for f in removed_features:
    print(f" - {f}")

# =========================================================
# SAVE TO EXCEL
# =========================================================
with pd.ExcelWriter(
    excel_path,
    mode="a",
    engine="openpyxl",
    if_sheet_exists="replace"
) as writer:

    # Sheet 1 -> Combined Report
    combined_report.to_excel(
        writer,
        sheet_name="RFE_Report",
        index=False
    )

    # Sheet 2 -> Selected Data
    df[selected_features + [target_column]].to_excel(
        writer,
        sheet_name="Selected_Data_RFE",
        index=False
    )

open_excel_file(excel_path)
print("\n[+] RFE results successfully saved to 'RFE_Report' and 'Selected_Data_RFE' in Data.xlsx")