import pandas as pd
import numpy as np
import os
import win32com.client
import warnings
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Import only the 3 specific models you requested
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor

# Suppress minor warnings for a clean console
warnings.filterwarnings('ignore')

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
sheet_name = "KfoldXGBR"  # Change this to your actual sheet name if different

print(f"Loading data from sheet: '{sheet_name}'...")
df = pd.read_excel(filepath, sheet_name=sheet_name)
target_column = df.columns[-1]
X_full = df.drop(columns=[target_column])
y = df[target_column]

# ================== Models & Hyperparameter Grids (For Nested CV) ==================
models_and_params = {
    "ENR": {
        "model": ElasticNet(random_state=42),
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.9]
        }
    },
    "XGBR": {
        "model": XGBRegressor(random_state=42, objective='reg:squarederror'),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "ETR": {
        "model": ExtraTreesRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    }
}

print("✅ Models ready for Nested Cross-Validation (ENR, XGBR, ETR)")

# ================== Nested K-Fold Execution ==================
# Outer CV: Tests the model on 5 chunks sequentially (unshuffled)
outer_cv = KFold(n_splits=5, shuffle=False)

# Inner CV: Tunes the hyperparameters on the training data using 3 folds
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

metrics_df_dict = {}
df_reordered_dict = {}
fold_indices_dict = {}

for model_name, config in models_and_params.items():
    fold_metrics_list = []
    fold_indices_list = []

    print(f"\n🚀 Processing {model_name} with Nested CV...")

    # Outer Loop
    for fold_index, (train_idx, test_idx) in enumerate(outer_cv.split(X_full), 1):
        X_train = X_full.iloc[train_idx]
        X_test  = X_full.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner Loop (Grid Search for hyperparameter tuning)
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1 # Uses all CPU cores to speed up the grid search
        )
        
        # Fit GridSearch on Train Data (Finds best params for this specific fold)
        grid_search.fit(X_train, y_train)
        
        # Predict on Test Data using the Best Estimator found by the inner loop
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calculate Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        fold_metrics_list.append({
            "Fold": fold_index,
            "R2": r2,
            "RMSE": rmse,
            "Best Inner Params": str(grid_search.best_params_) # Records what params won the inner loop
        })

        fold_indices_list.append({
            "train_idx": train_idx,
            "test_idx": test_idx
        })
        
        print(f"   - Fold {fold_index}: R2 = {r2:.4f} | RMSE = {rmse:.4f}")

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
for model_name in models_and_params.keys():
    metrics_df = metrics_df_dict[model_name]
    best_fold = metrics_df.loc[metrics_df["R2"].idxmax()]

    summary_rows.append({
        "Model": model_name,
        "Best Fold": best_fold["Fold"],
        "Best Inner Params": best_fold["Best Inner Params"],
        "Best R2": best_fold["R2"],
        "Best RMSE": best_fold["RMSE"],
        "Mean R2": metrics_df["R2"].mean(),
        "Mean RMSE": metrics_df["RMSE"].mean()
    })

summary_df = pd.DataFrame(summary_rows)

# ================== Save to Excel ==================
# Make sure the file is closed before attempting to write to it!
close_excel_file(filepath)

print("\nSaving results to Excel...")
with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for model_name in models_and_params.keys():
        metrics_df_dict[model_name].to_excel(
            writer, sheet_name=f"{model_name}_Nested_Metrics", index=False
        )
        df_reordered_dict[model_name].to_excel(
            writer, sheet_name=f"Data_after_Nested_{model_name}", index=False
        )
    summary_df.to_excel(writer, sheet_name="Nested_Model_Summary", index=False)

open_excel_file(filepath)

# ================== Print Results ==================
print("\n" + "="*80)
print(" NESTED CROSS-VALIDATION SUMMARY ".center(80))
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)