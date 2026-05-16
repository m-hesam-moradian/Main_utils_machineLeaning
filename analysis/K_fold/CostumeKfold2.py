import pandas as pd
import numpy as np
import os
import win32com.client
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# --- Scaling Requirement for Distance Models ---
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- New Model Imports (The "Perfect 6") ---
from sklearn.linear_model import ElasticNet
from interpret.glassbox import ExplainableBoostingRegressor # pip install interpret
from xgboost import XGBRegressor                          # pip install xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

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
sheet_name = "DATA_Shuffled"  # Change this to your actual sheet name if different

df = pd.read_excel(filepath, sheet_name=sheet_name)
target_column = df.columns[-1]
X_full = df.drop(columns=[target_column])
y = df[target_column]

# ================== Updated Models (The 6 Paradigms) ==================
models = {
    # 1. Linear Model
    "ElasticNet": ElasticNet(
    # alpha=0.2,
    # l1_ratio=0.7,
    max_iter=10,
    # tol=1e-4,
    # random_state=42
    ),
    
    # 2. Generalized Additive Model (GAM)
    "EBM": ExplainableBoostingRegressor(
    interactions=0,        # pure GAM (fully interpretable)
    max_bins=200,
    learning_rate=0.006,
    max_rounds=400
    ),
    
    # 3. Advanced Tree-Based Ensemble (Boosting)
    "XGBoost": XGBRegressor(
    n_estimators=7,       # keep moderate
    # max_depth=4,            # smaller depth
    # learning_rate=0.05,     # slower learning
    # subsample=0.8,          # randomness to reduce overfit
    # colsample_bytree=0.8,
    # reg_alpha=0.1,
    # reg_lambda=1
    ),
    
    # 4. Advanced Tree-Based Ensemble (Bagging)
    "RandomForest": RandomForestRegressor(
    n_estimators=10,
    max_depth=5,
    # min_samples_leaf=10,
    # max_features=0.7,
    # random_state=42,
    # n_jobs=-1
    ),
    
    # 5. Mathematical Margin Model (Needs Scaling)
    "SVR_RBF": SVR(
    kernel="rbf",   # non-linear regression
    # C=70,          # regularization strength
    # epsilon=0.1,    # epsilon-insensitive loss
    # gamma="scale"   # kernel coefficient
    # tol=1e-3, 
    max_iter=1500
),
    
    # 6. Distance / Neural Model (Needs Scaling, increased max_iter for convergence)
    "MLP_Neural": MLPRegressor(
    hidden_layer_sizes=(100,),  # single hidden layer with 100 neurons
    activation='relu',
    solver='adam',
    # alpha=0.0001,
    max_iter=100,  # increase iterations for better convergence
    random_state=42
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
    fold_indices_list =[]

    print(f"Processing {model_name}...")

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X_full), 1):
        X_train = X_full.iloc[train_idx]
        X_test  = X_full.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        
        # --- Make Predictions ---
        y_pred = model.predict(X_test)
        
        # ✅ --- ROUNDING STEP (IMPORTANT) ---
        y_pred = np.round(y_pred).astype(int)

        # --- Calculate Metrics ---
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
summary_rows =[]
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
print("\n" + "="*50)
print(summary_df.to_string(index=False))
print("="*50)