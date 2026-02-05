import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from lightgbm import LGBMRegressor  # Swapped from LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    r2_score, mean_squared_error
)

# --- Excel helpers (close_excel_file and open_excel_file remain the same) ---
def close_excel_file(filepath):
    import os
    import win32com.client
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("💾 Saved and 🔒 Closed Excel file:", filepath)
                break
    except Exception:
        pass

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
X_np = np.array(X)
y_np = np.array(y)

# --- Detect task type ---
is_classification = False
if pd.api.types.is_integer_dtype(y_np) and len(np.unique(y_np)) <= 20:
    is_classification = True

print("Task type:", "Classification" if is_classification else "Regression (LGBM)")

# --- Train/test split ---
x_train, x_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, shuffle=True, random_state=42,
    stratify=y_np if is_classification else None
)

# --- Feature Selection ---
if is_classification:
    estimator = LinearSVC(random_state=42, max_iter=5000)
else:
    # verbosity=-1 silences info logs during the RFE iterations
    estimator = LGBMRegressor(random_state=42, verbosity=-1)

selector = RFE(estimator=estimator, n_features_to_select=1)  # rank all features
selector.fit(x_train, y_train)

feature_ranking = selector.ranking_
ranking_df = pd.DataFrame({"Feature": col, "Rank": feature_ranking})

# --- Select features based on rank threshold ---
selected_features = ranking_df[ranking_df["Rank"] <= 9]["Feature"].tolist()

# --- Evaluate subsets ---
report_rows = []
input_and_ranks = [{"input": j, "rank": feature_ranking[j]} for j in range(len(feature_ranking))]
input_and_ranks.sort(key=lambda x: x["rank"])
new_sort_X = X_np[:, [x["input"] for x in input_and_ranks]]



for input_count in range(1, new_sort_X.shape[1] + 1):
    X_subset = new_sort_X[:, :input_count]
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_subset, y_np, test_size=0.2, shuffle=True, random_state=42,
        stratify=y_np if is_classification else None
    )
    
    if is_classification:
        model = LinearSVC(random_state=42, max_iter=5000)
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te)
        acc = accuracy_score(Y_te, pred)
        f1 = f1_score(Y_te, pred, average="weighted")
        report_rows.append({"Features used": input_count, "Accuracy": acc, "F1-Score": f1})
    else:
        # Evaluate with LGBM
        model = LGBMRegressor(random_state=42, verbosity=-1)
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te)
        r2 = r2_score(Y_te, pred)
        rmse = np.sqrt(mean_squared_error(Y_te, pred))
        report_rows.append({"Features used": input_count, "R2": r2, "RMSE": rmse})

# --- Save results to Excel ---
report_df = pd.DataFrame(report_rows)

# Using 'with' handles closing the writer automatically
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists='replace') as writer:
    report_df.to_excel(writer, sheet_name="RFE_Report", index=False)
    ranking_df.to_excel(writer, sheet_name="Feature_Ranking", index=False)
    df[selected_features + [target_column]].to_excel(
        writer, sheet_name="Selected_Data", index=False
    )

open_excel_file(excel_path)
print("✅ RFE results saved to Excel using LGBM for Regression.")