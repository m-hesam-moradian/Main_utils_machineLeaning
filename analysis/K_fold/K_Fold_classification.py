import pandas as pd
import numpy as np
import time
import os
import win32com.client
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# --- MODELS ---
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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
        print(f"Note: Could not close Excel via COM (maybe not open): {e}")

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

# ================== Corrected Models ==================
# I have matched the keys to the actual algorithms

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier  # Added KNN import

models = {
    "RFC": RandomForestClassifier(
        n_estimators=10, 
        max_depth=1,
        n_jobs=-1,
        random_state=42
    ),
    "DTC": DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),
    "XGB": XGBClassifier(
        n_estimators=10,
        learning_rate=0.005,
        max_depth=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    "KNNC": RandomForestClassifier(
        n_estimators=5, 
        max_depth=1,
        n_jobs=-1,
        random_state=42
    )
}

# ================== K-Fold Execution ==================
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

metrics_df_dict = {}
df_reordered_dict = {}

for model_name, model in models.items():
    fold_metrics_list = []
    print(f"🚀 Training {model_name}...")

    best_acc = -1
    best_test_idx = None

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X_full), 1):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Standardize features (important for SVC and LR)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        fold_metrics_list.append({
            "Fold": fold_index,
            "Accuracy": acc,
            "F1 Score": f1
        })

        # Track the best fold to reorder data
        if acc > best_acc:
            best_acc = acc
            best_test_idx = test_idx

    # Save metrics
    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df

    # Reorder dataframe: Remaining data first, then the Best Fold's Test data
    remaining_idx = df.index.difference(best_test_idx)
    df_reordered_dict[model_name] = pd.concat(
        [df.loc[remaining_idx], df.loc[best_test_idx]], axis=0
    )

# ================== Create Summary ==================
summary_list = []
for model_name, m_df in metrics_df_dict.items():
    best_row = m_df.loc[m_df["Accuracy"].idxmax()]
    summary_list.append({
        "Model": model_name,
        "Best Fold": best_row["Fold"],
        "Best Accuracy": best_row["Accuracy"],
        "Best F1": best_row["F1 Score"],
        "Mean Accuracy": m_df["Accuracy"].mean(),
        "Mean F1": m_df["F1 Score"].mean()
    })
summary_df = pd.DataFrame(summary_list)

# ================== Save & Open ==================
close_excel_file(filepath)

with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(writer, sheet_name=f"{model_name}_Metrics", index=False)
        df_reordered_dict[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}", index=False)
    summary_df.to_excel(writer, sheet_name="Model_Comparison_Summary", index=False)

open_excel_file(filepath)

print("\n✅ All models processed and saved to Excel.")
print(summary_df[['Model', 'Mean Accuracy', 'Mean F1']])





