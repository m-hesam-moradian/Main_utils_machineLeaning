import pandas as pd
import os
import win32com.client
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.svm import SVR  # Standard implementation for SVR

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
    except Exception as e:
        print(f"Note: Could not close Excel via COM: {e}")

def open_excel_file(filepath):
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("📂 Opened Excel file:", filepath)
    except Exception as e:
        print(f"Could not open Excel automatically: {e}")

# ================== Load Dataset ==================
filepath = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_After_ANOVA"  # Adjusted to match previous steps (change if needed)
# sheet_name = "Data_after_MRMRMS"  # Adjusted to match previous steps (change if needed)

df = pd.read_excel(filepath, sheet_name=sheet_name)
target_column = df.columns[-1]
X_full = df.drop(columns=[target_column])
y = df[target_column]

# Note: Removed pd.qcut() because Regression requires continuous target values.

# ================== Models ==================
models = {
    # Using parameters based on the paper's ε-SVR configuration
    "LSSVR": SVR(
        kernel='rbf', 
        C=0.001, 
        gamma='scale', 
        epsilon=0.1
    )
}

# ================== K-Fold ==================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

metrics_df_dict = {}
df_reordered_dict = {}

for model_name, model in models.items():
    fold_metrics_list = []
    print(f"🚀 Training {model_name}...")

    best_score = -np.inf  # Track highest R2
    best_test_idx = None

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X_full), 1):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train & Predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ✅ Regression Metrics
        r2 = r2_score(y_test, y_pred)
        # Multiply by 100 to get traditional MAPE percentage (e.g., 5.4%)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100 

        fold_metrics_list.append({
            "Fold": fold_index,
            "R2": r2,
            "MAPE (%)": mape
        })

        # Best fold is based on the highest R2 score
        if r2 > best_score:
            best_score = r2
            best_test_idx = test_idx

    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df

    # Reorder dataset to put the best fold data at the bottom
    remaining_idx = df.index.difference(best_test_idx)
    df_reordered_dict[model_name] = pd.concat(
        [df.loc[remaining_idx], df.loc[best_test_idx]], axis=0
    )

# ================== Summary ==================
summary_list = []
for model_name, m_df in metrics_df_dict.items():
    # Find the row with the highest R2 score
    best_row = m_df.loc[m_df["R2"].idxmax()]

    summary_list.append({
        "Model": model_name,
        "Best Fold": best_row["Fold"],
        "Best R2": best_row["R2"],
        "Best MAPE (%)": best_row["MAPE (%)"],
        "Mean R2": m_df["R2"].mean(),
        "Mean MAPE (%)": m_df["MAPE (%)"].mean()
    })

summary_df = pd.DataFrame(summary_list)

# Rank based on Mean R2 (Higher is better)
summary_df["Rank"] = summary_df["Mean R2"].rank(ascending=False).astype(int)
summary_df = summary_df.sort_values("Rank")

# ================== Save ==================
close_excel_file(filepath)

with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(writer, sheet_name=f"{model_name}_Metrics(ANOVA)", index=False)
        df_reordered_dict[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}(ANOVA)", index=False)
    # summary_df.to_excel(writer, sheet_name="Model_Comparison_Summary", index=False)

open_excel_file(filepath)

print("\n✅ Models processed (R2 & MAPE):")
print(summary_df)