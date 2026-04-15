import pandas as pd
import os
import win32com.client
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier  # NEW

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
sheet_name = "data_after_chi2"

df = pd.read_excel(filepath, sheet_name=sheet_name)
target_column = df.columns[-1]
X_full = df.drop(columns=[target_column])
y = df[target_column]

# Ensure classification labels
if y.dtype != 'int64' and y.dtype != 'object':
    y = pd.qcut(y, q=3, labels=False)

# ================== Models ==================
models = {
    "QDA": QuadraticDiscriminantAnalysis(
        reg_param=0.9,
        store_covariance=True,
        tol=1e-4
    ),
    "LGBC": LGBMClassifier(
        n_estimators=5,
        learning_rate=0.0024,
        max_depth=2,
        random_state=42
    )
}

# ================== K-Fold ==================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

metrics_df_dict = {}
df_reordered_dict = {}

for model_name, model in models.items():
    fold_metrics_list = []
    print(f"🚀 Training {model_name}...")

    best_score = -1
    best_test_idx = None

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X_full), 1):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ✅ New Metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        fold_metrics_list.append({
            "Fold": fold_index,
            "Accuracy": accuracy,
            "Recall": recall
        })

        # Best fold based on average score
        score = (accuracy + recall) / 2
        if score > best_score:
            best_score = score
            best_test_idx = test_idx

    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df

    remaining_idx = df.index.difference(best_test_idx)
    df_reordered_dict[model_name] = pd.concat(
        [df.loc[remaining_idx], df.loc[best_test_idx]], axis=0
    )

# ================== Summary ==================
summary_list = []
for model_name, m_df in metrics_df_dict.items():
    best_row = m_df.loc[((m_df["Accuracy"] + m_df["Recall"]) / 2).idxmax()]

    summary_list.append({
        "Model": model_name,
        "Best Fold": best_row["Fold"],
        "Best Accuracy": best_row["Accuracy"],
        "Best Recall": best_row["Recall"],
        "Mean Accuracy": m_df["Accuracy"].mean(),
        "Mean Recall": m_df["Recall"].mean()
    })

summary_df = pd.DataFrame(summary_list)

summary_df["Score"] = (summary_df["Mean Accuracy"] + summary_df["Mean Recall"]) / 2
summary_df["Rank"] = summary_df["Score"].rank(ascending=False).astype(int)
summary_df = summary_df.sort_values("Rank")

# ================== Save ==================
close_excel_file(filepath)

with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(writer, sheet_name=f"{model_name}_Metrics(CHI2)", index=False)
        df_reordered_dict[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}(CHI2)", index=False)
    summary_df.to_excel(writer, sheet_name="Model_Comparison_Summary(CHI2)", index=False)

open_excel_file(filepath)

print("\n✅ Models processed (Accuracy & Recall):")
print(summary_df)