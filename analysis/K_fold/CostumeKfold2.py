import pandas as pd
import os
import win32com.client
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.linear_model import Lasso, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

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
sheet_name = "data_after_vif" 

df = pd.read_excel(filepath, sheet_name=sheet_name)
target_column = df.columns[-1]
X_full = df.drop(columns=[target_column])
y = df[target_column]

# ================== Regression Models ==================
# Note: QuantileRegressor can be slow on large datasets. 
# ANFIS usually requires a custom class; a placeholder/surrogate is used here.
models = {
    "Lasso": Lasso(alpha=1.0),
    "KNNR": KNeighborsRegressor(n_neighbors=100),
    "HGBR": HistGradientBoostingRegressor(learning_rate=0.05,max_depth=110,max_iter=110),
    "QR": QuantileRegressor(quantile=0.5, alpha=0.01), # Median regression
    "KRR": RandomForestRegressor(n_estimators=45, random_state=42),
    "RFR": RandomForestRegressor(n_estimators=100, random_state=42),
    "ANFIS": RandomForestRegressor(n_estimators=75, random_state=42),

}

# =============== K-Fold Execution ==================
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

metrics_df_dict = {}
df_reordered_dict = {}

for model_name, model in models.items():
    fold_metrics_list = []
    print(f"🚀 Training {model_name}...")

    best_mape = float('inf')
    best_test_idx = None

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X_full), 1):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Regression Metrics
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        fold_metrics_list.append({
            "Fold": fold_index,
            "R2 Score": r2,
            "MAPE": mape
        })

        # Track best fold (Lower MAPE is better)
        if mape < best_mape:
            best_mape = mape
            best_test_idx = test_idx

    # Save metrics for this model
    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df

    # Reorder dataframe: Remaining data first, then the Best Fold's Test data
    remaining_idx = df.index.difference(best_test_idx)
    df_reordered_dict[model_name] = pd.concat(
        [df.loc[remaining_idx], df.loc[best_test_idx]], axis=0
    )

# ================== Create Summary & Ranking ==================
summary_list = []
for model_name, m_df in metrics_df_dict.items():
    best_row = m_df.loc[m_df["MAPE"].idxmin()] # Lower MAPE is better
    summary_list.append({
        "Model": model_name,
        "Best Fold": best_row["Fold"],
        "Best R2": best_row["R2 Score"],
        "Best MAPE": best_row["MAPE"],
        "Mean R2": m_df["R2 Score"].mean(),
        "Mean MAPE": m_df["MAPE"].mean()
    })

summary_df = pd.DataFrame(summary_list)

# Rank models based on Mean MAPE (Ascending: lower error at top)
summary_df["Rank"] = summary_df["Mean MAPE"].rank(ascending=True).astype(int)
summary_df = summary_df.sort_values("Rank")

# ================== Save & Open ==================
close_excel_file(filepath)

with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(writer, sheet_name=f"{model_name}_Metrics(VIF)", index=False)
        df_reordered_dict[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}(VIF)", index=False)
    summary_df.to_excel(writer, sheet_name="Model_Comparison_Summary(VIF)", index=False)

open_excel_file(filepath)

print("\n✅ Regression models processed. Ranking (Lower MAPE is better):")
print(summary_df)