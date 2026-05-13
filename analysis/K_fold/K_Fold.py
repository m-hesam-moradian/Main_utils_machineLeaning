import pandas as pd
import numpy as np
import os
import win32com.client
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error


# ================== Excel Helpers ==================
def close_excel_file(filepath):
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("💾 Saved and 🔒 Closed Excel file:", filepath)
                break
        excel.Quit()
    except Exception:
        pass

def open_excel_file(filepath):
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("📂 Opened Excel file:", filepath)
    except Exception:
        pass

# ================== Load Dataset ==================
filepath = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Encoded_Data"  # Change this to your actual sheet name if different

df = pd.read_excel(filepath, sheet_name=sheet_name)
target_column = df.columns[-1]
X_full = df.drop(columns=[target_column])
y = df[target_column]

# ================== Updated Models ==================
from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor

models = {

    "ENR": ElasticNet(
        # alpha=1.0,        # Regularization strength
        # # l1_ratio=0.5,     # Balance between L1 and L2 regularization
        # max_iter=1000,
        # random_state=42
    ),

    "CATR": CatBoostRegressor(
        # iterations=500,
        # learning_rate=0.05,
        # depth=6,
        # loss_function='RMSE',
        # verbose=0,
        # random_state=42
    )
}

print("✅ Models ready for training:")
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

        fold_metrics_list.append({
            "Fold": fold_index,
            "R2": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        })

        fold_indices_list.append({
            "train_idx": train_idx,
            "test_idx": test_idx
        })

    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df
    fold_indices_dict[model_name] = fold_indices_list

    # Reorder data based on best fold
    best_fold_idx = metrics_df["R2"].idxmax()
    best_test_idx = fold_indices_dict[model_name][best_fold_idx]["test_idx"]
    remaining_idx = df.index.difference(best_test_idx)
    
    df_reordered_dict[model_name] = pd.concat(
        [df.loc[remaining_idx], df.loc[best_test_idx]], axis=0
    )

# ================== Summary Generation ==================
summary_rows = []
for model_name in models:
    metrics_df = metrics_df_dict[model_name]
    best_fold = metrics_df.loc[metrics_df["R2"].idxmax()]

    summary_rows.append({
        "Model": model_name,
        "Best Fold": best_fold["Fold"],
        "Best R2": best_fold["R2"],
        "Best RMSE": best_fold["RMSE"],
        "Mean R2": metrics_df["R2"].mean(),
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
print("\n" + "="*30)
print(summary_df.to_string(index=False))
print("="*30)