import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

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
# LOAD DATA (From ENN_Data)
# =========================================================
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

df = pd.read_excel(excel_path, sheet_name="ENN_Data")

target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

col_names = X.columns
X_np = np.array(X)
y_np = np.array(y)

# Standardize features for accurate linear ranking
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled,
    y_np,
    test_size=0.2,
    shuffle=True,
    random_state=42,
    stratify=y_np
)

# =========================================================
# FEATURE SELECTION (Lighter, 1-by-1 step)
# =========================================================
estimator = LinearSVC(
    random_state=42,
    max_iter=10000,
    C=1.0
)

selector = RFE(
    estimator=estimator,
    n_features_to_select=1,
    step=1
)

selector.fit(x_train, y_train)
feature_ranking = selector.ranking_

ranking_df = pd.DataFrame({
    "Feature": col_names,
    "Rank": feature_ranking
}).sort_values("Rank").reset_index(drop=True)

# Lighter cutoff: Retain top 18 features (out of 43)
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
# EVALUATE SUBSETS (Progressive curve)
# =========================================================
report_rows = []

# Order features by rank
sorted_feature_indices = np.argsort(selector.ranking_)

for input_count in range(1, X_scaled.shape[1] + 1):
    sub_indices = sorted_feature_indices[:input_count]
    X_subset = X_scaled[:, sub_indices]

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_subset,
        y_np,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y_np
    )

    model = LinearSVC(
        random_state=42,
        max_iter=10000,
        C=1.0
    )
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_te)

    acc = accuracy_score(Y_te, pred)
    f1 = f1_score(Y_te, pred, average="weighted")

    report_rows.append({
        "Features used": input_count,
        "Accuracy": acc,
        "F1-Score": f1
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