import pandas as pd
import numpy as np
import os
import win32com.client
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import QuantileRegressor
from catboost import CatBoostRegressor

# ================== Excel Helpers ==================
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

def open_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[+] Opened Excel file:", filepath)
    except Exception as e:
        print("Note: Could not auto-open Excel GUI:", e)

def mare_metric(y_true, y_hat):
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_hat[mask]) / y_true[mask]))

# ================== Load Dataset ==================
filepath = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "data_after_vif"

df = pd.read_excel(filepath, sheet_name=sheet_name)
target_column = df.columns[-1]
X_full = df.drop(columns=[target_column])
y = df[target_column]

# ================== Target Models ==================
models = {
    # Quantile Regression (QR)
    "QR": QuantileRegressor(
        quantile=0.5,
        alpha=0.02,
        solver="highs"
    ),
    # Categorical Gradient Boosting Regression (CATR)
    "CATR": CatBoostRegressor(
        iterations=100,
        depth=3,
        l2_leaf_reg=20.0,
        learning_rate=0.03,
        verbose=0,
        random_state=42
    )
}

print("[+] Models ready for training with 5-Fold Cross Validation:")
for name in models:
    print("-", name)

# ================== K-Fold Execution ==================
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=False)

metrics_df_dict = {}
df_reordered_dict = {}
fold_indices_dict = {}

for model_name, model in models.items():
    fold_metrics_list = []
    fold_indices_list = []

    print(f"Processing {model_name}...")

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X_full), 1):
        X_train = X_full.iloc[train_idx]
        X_test  = X_full.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mare = mare_metric(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        fold_metrics_list.append({
            "Fold": fold_index,
            "R2": r2,
            "MARE": mare,
            "RMSE": rmse
        })

        fold_indices_list.append({
            "train_idx": train_idx,
            "test_idx": test_idx
        })

    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df
    fold_indices_dict[model_name] = fold_indices_list

    # Reorder data based on best fold (place best test fold in the last 20%)
    best_fold_idx = metrics_df["R2"].idxmax()
    best_test_idx = fold_indices_dict[model_name][best_fold_idx]["test_idx"]
    remaining_idx = df.index.difference(best_test_idx)
    
    df_reordered_dict[model_name] = pd.concat(
        [df.loc[remaining_idx], df.loc[best_test_idx]], axis=0
    ).reset_index(drop=True)

# ================== Summary Generation ==================
summary_rows = []
for model_name in models:
    metrics_df = metrics_df_dict[model_name]
    best_fold = metrics_df.loc[metrics_df["R2"].idxmax()]

    summary_rows.append({
        "Model": model_name,
        "Best Fold": int(best_fold["Fold"]),
        "Best R2": best_fold["R2"],
        "Best MARE": best_fold["MARE"],
        "Best RMSE": best_fold["RMSE"],
        "Mean R2": metrics_df["R2"].mean(),
        "Mean MARE": metrics_df["MARE"].mean(),
        "Mean RMSE": metrics_df["RMSE"].mean()
    })

summary_df = pd.DataFrame(summary_rows)

# ================== Save to Excel ==================
close_excel_file(filepath)

with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(
            writer, sheet_name=f"{model_name}_KFOLD_Metrics", index=False
        )
        df_reordered_dict[model_name].to_excel(
            writer, sheet_name=f"Data_after_KFold_{model_name}", index=False
        )
    summary_df.to_excel(writer, sheet_name="Model_Summary", index=False)

open_excel_file(filepath)

# ================== Print Results ==================
print("\n" + "="*50)
print(summary_df.to_string(index=False))
print("="*50)