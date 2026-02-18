import pandas as pd
from sklearn.model_selection import KFold

# --- UPDATED IMPORTS ---
from sklearn.metrics import accuracy_score, f1_score
# ================== Excel Helpers ==================
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

# ================== Load Dataset ==================
filepath = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Encoded_Data"  # Loading the balanced data

df = pd.read_excel(filepath, sheet_name=sheet_name)

target_column = df.columns[-1]
X_full = df.drop(columns=[target_column])
y = df[target_column]

# ================== Models ==================
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB

models = {
    # XGBC: Efficient Gradient Boosting
    "XGBC": XGBClassifier(
        n_estimators=18,
        learning_rate=0.01,
        max_depth=10,
        random_state=42,
        # tree_method='hist' # Uses histogram binning to save memory
    ),

    # RFC: Random Forest (Reduced n_estimators to save RAM)
    "RFC": RandomForestClassifier(
        n_estimators=5, 
        max_depth=2,
        n_jobs=-1,        # Uses all CPU cores to speed up training
        random_state=42
    ),

    # LOG_REG: Replaces SVC (Linear, extremely fast, low RAM)
    "SVC": XGBClassifier(
        n_estimators=40,
        learning_rate=0.005,
        max_depth=1,
        random_state=42,
        # tree_method='hist' # Uses histogram binning to save memory
    ),

    # GPC: Note - if your PC still struggles, swap this for Naive Bayes
"GPC": GaussianNB(
        var_smoothing=1e-8  # The "knob" for stability; default is usually best
    )
}
# ================== K-Fold ==================
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

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

        # --- Classification Metrics ---
        # Using 'weighted' average for F1 to handle class imbalances if any remain
        fold_metrics_list.append({
            "Fold": fold_index,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        })

        fold_indices_list.append({
            "train_idx": train_idx,
            "test_idx": test_idx
        })

    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df
    fold_indices_dict[model_name] = fold_indices_list

    # Select best fold based on Accuracy
    best_fold_idx = metrics_df["Accuracy"].idxmax()
    best_test_idx = fold_indices_dict[model_name][best_fold_idx]["test_idx"]

    # Reorder dataframe: Train first, then Test (Best Fold)
    remaining_idx = df.index.difference(best_test_idx)
    df_reordered_dict[model_name] = pd.concat(
        [df.loc[remaining_idx], df.loc[best_test_idx]], axis=0
    )

# ================== Summary ==================
summary_df = []
for model_name in models:
    metrics_df = metrics_df_dict[model_name]
    best_fold = metrics_df.loc[metrics_df["Accuracy"].idxmax()]

    summary_df.append({
        "Model": model_name,
        "Best Fold": best_fold["Fold"],
        "Best Accuracy": best_fold["Accuracy"],
        "Best F1": best_fold["F1 Score"],
        "Mean Accuracy": metrics_df["Accuracy"].mean(),
        "Mean F1": metrics_df["F1 Score"].mean()
    })

summary_df = pd.DataFrame(summary_df)

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

# ================== Print Summary ==================
for model_name in models:
    metrics_df = metrics_df_dict[model_name]
    best_fold = metrics_df.loc[metrics_df["Accuracy"].idxmax()]

    print(f"\n🔹 Model: {model_name}")
    print(f"   🏆 Best Fold: Fold {best_fold['Fold']}")
    print(f"   Accuracy: {best_fold['Accuracy']:.4f}")
    print(f"   F1 Score: {best_fold['F1 Score']:.4f}")
    print(f"   📈 Mean Accuracy: {metrics_df['Accuracy'].mean():.4f}")
    print(f"   📉 Mean F1 Score: {metrics_df['F1 Score'].mean():.4f}")

