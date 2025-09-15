import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# --- Load dataset ---
file_path = (
    r"D:\ML\Main_utils\Task\Original Dataset- Concrete (elevated temperature).xlsx"
)
sheet_name = "Standard_Normalized"
target_col = "Compressive Strength"  # replace with your target column

df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()

# Features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# --- Define models ---
models = {
    "GPR": GaussianProcessRegressor(),
    "GBR": GradientBoostingRegressor(random_state=42),
}

# --- K-Fold setup ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=False)

# Dictionaries to store results
metrics_df_dict = {}  # Stores metrics table for each model
df_reordered_dict = {}  # Stores reordered dataset for each model
fold_indices_dict = {}  # Stores fold indices for each model

# --- Loop through models ---
for model_name, model in models.items():
    print(f"\n--- Evaluating {model_name} ---")

    fold_metrics_list = []
    fold_indices_list = []

    # K-Fold CV
    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Save metrics
        fold_metrics_list.append({"Fold": fold_index, "R2": r2, "RMSE": rmse})
        fold_indices_list.append({"train_idx": train_idx, "test_idx": test_idx})

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df
    fold_indices_dict[model_name] = fold_indices_list

    # Identify best fold based on RMSE
    best_fold_idx = metrics_df["RMSE"].idxmin()  # index of best fold
    best_test_idx = fold_indices_list[best_fold_idx]["test_idx"]

    # Reorder dataset: move best fold to the end
    remaining_idx = df.index.difference(best_test_idx)
    df_reordered = pd.concat([df.loc[remaining_idx], df.loc[best_test_idx]], axis=0)
    df_reordered_dict[model_name] = df_reordered

# --- OUTPUT ---
# Now you have two tables per model
# metrics_df_dict[model_name] → fold metrics
# df_reordered_dict[model_name] → reordered dataset

# Example: print metrics tables
for model_name, metrics_df in metrics_df_dict.items():
    print(f"\nMetrics table for {model_name}:")
    print(metrics_df)

# Example: print head of reordered datasets
for model_name, df_reordered in df_reordered_dict.items():
    print(f"\nReordered dataset for {model_name} (best fold moved to end):")
    print(df_reordered.head())  # just first 5 rows
