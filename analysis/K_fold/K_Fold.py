import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error


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

# --- Load dataset ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Encoded_Data"
close_excel_file(excel_path)
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Define models ---
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "LGBc": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42
    ),
    "RFc": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
}

# --- K-Fold setup ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=False)

# Dictionaries to store results
metrics_df_dict = {}
df_reordered_dict = {}
fold_indices_dict = {}

# --- Loop through models ---
for model_name, model in models.items():
    fold_metrics_list = []
    fold_indices_list = []

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        fold_metrics_list.append({
            "Fold": fold_index,
            "R2": r2,
            "RMSE": rmse
        })
        fold_indices_list.append({"train_idx": train_idx, "test_idx": test_idx})

    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df
    fold_indices_dict[model_name] = fold_indices_list

    best_fold_idx = metrics_df["R2"].idxmax()
    best_test_idx = fold_indices_dict[model_name][best_fold_idx]["test_idx"]

    remaining_idx = df.index.difference(best_test_idx)
    df_reordered = pd.concat([df.loc[remaining_idx], df.loc[best_test_idx]], axis=0)
    df_reordered_dict[model_name] = df_reordered

# --- Summary Table ---
summary_df = []
for model_name in models:
    metrics_df = metrics_df_dict[model_name]
    best_fold_idx = metrics_df["R2"].idxmax()
    best_fold = metrics_df.loc[best_fold_idx]

    summary_df.append({
        "Model": model_name,
        "Best Fold": best_fold["Fold"],
        "Best R2": best_fold["R2"],
        "Best RMSE": best_fold["RMSE"],
        "Mean R2": metrics_df["R2"].mean(),
        "Mean RMSE": metrics_df["RMSE"].mean(),
    })

# --- Save to Excel ---
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(writer, sheet_name=f"{model_name}_KFOLD_Metrics", index=False)
        df_reordered_dict[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}", index=False)
    pd.DataFrame(summary_df).to_excel(writer, sheet_name="Model_Summary", index=False)



open_excel_file(excel_path)

# --- Print Summary ---
for model_name in models:

    metrics_df = metrics_df_dict[model_name]
    best_fold_idx = metrics_df["R2"].idxmax()
    best_fold = metrics_df.loc[best_fold_idx]

    print(f"\n🔹 Model: {model_name}")
    print(f"   🏆 Best Fold: Fold {best_fold['Fold']}")
    print(f"   R2: {best_fold['R2']:.4f}")
    print(f"   RMSE: {best_fold['RMSE']:.4f}")
    print(f"   📈 Mean R2: {metrics_df['R2'].mean():.4f}")
    print(f"   📉 Mean RMSE: {metrics_df['RMSE'].mean():.4f}")