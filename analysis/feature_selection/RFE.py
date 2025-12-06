import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --- Excel helpers ---
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

# --- Load data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)
df = pd.read_excel(excel_path, sheet_name="Data")

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

col = X.columns
X = np.array(X)
y = np.array(y)

# --- Train/test split ---
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
)

# --- Feature Selection with RFE (SVC) ---
desired_features = 4
svc_estimator = SVC(kernel="linear", random_state=42)
selector = RFE(estimator=svc_estimator, n_features_to_select=desired_features)
selector.fit(x_train, y_train)

selected_features = col[selector.support_]
feature_ranking = selector.ranking_

print("Selected features:", selected_features)
print("Feature ranking:", feature_ranking)

# --- Evaluate subsets ---
report_rows = []
input_and_ranks = [{"input": j, "rank": feature_ranking[j]} for j in range(len(feature_ranking))]
input_and_ranks.sort(key=lambda x: x["rank"])
new_sort_X = X[:, [x["input"] for x in input_and_ranks]]

for input in range(1, new_sort_X.shape[1] + 1):
    X_deleted = new_sort_X[:, :input]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_deleted, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(Y_test, pred)
    f1 = f1_score(Y_test, pred, average="weighted")
    report_rows.append({"Features used": input, "Accuracy": acc, "F1-Score": f1})

# --- Save results to Excel ---
report_df = pd.DataFrame(report_rows)
ranking_df = pd.DataFrame({"Feature": col, "Rank": feature_ranking})

with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl") as writer:
    report_df.to_excel(writer, sheet_name="RFE_Report", index=False)
    ranking_df.to_excel(writer, sheet_name="Feature_Ranking", index=False)
    df[selected_features.tolist() + [target_column]].to_excel(
        writer, sheet_name="Selected_Data", index=False
    )

open_excel_file(excel_path)