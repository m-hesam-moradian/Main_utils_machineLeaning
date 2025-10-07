import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error


# Make sure the class is in this path

# Import your EVOLUTIONARY_ANFIS class here


# --- Load dataset ---

sheet_name = "F-static"
df = pd.read_excel(
    r"D:\ML\Main_utils\task\startup_company_one_line_pitches.xlsx",
    sheet_name=sheet_name,
)
target_column = "Market_Size_Billion_USD"


# Features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Define models ---
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

models = {
    "SGB": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,  # stochastic element
        max_depth=3,
        random_state=42,
    ),
    "LGBR": LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
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

    fold_metrics_list = []
    fold_indices_list = []

    # K-Fold CV
    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit model
        if model_name == "ANFIS":
            model.fit(
                X_train.values, y_train.values
            )  # convert DataFrame/Series to ndarray
            y_pred = model.predict(
                X_test.values, *model.fit(X_train.values, y_train.values)
            )
        else:
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
summary_df = []

for model_name in models:
    metrics_df = metrics_df_dict[model_name]
    best_fold_idx = metrics_df["RMSE"].idxmin()
    best_fold = metrics_df.loc[best_fold_idx]

    summary_df.append(
        {
            "Model": model_name,
            "Best Fold": best_fold["Fold"],
            "Best R2": best_fold["R2"],
            "Best RMSE": best_fold["RMSE"],
            "Mean R2": metrics_df["R2"].mean(),
            "Mean RMSE": metrics_df["RMSE"].mean(),
        }
    )
excel_path = r"D:\ML\Main_utils\task\startup_company_one_line_pitches.xlsx"

with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(
            writer, sheet_name=f"{model_name}_KFOLD_Metrics", index=False
        )
        df_reordered_dict[model_name].to_excel(
            writer, sheet_name=f"Data_after_KFold_{model_name}", index=False
        )
    # pd.DataFrame(summary_df).to_excel(writer, sheet_name="Model_Summary", index=False)
for model_name in models:
    metrics_df = metrics_df_dict[model_name]
    best_fold_idx = metrics_df["RMSE"].idxmin()
    best_fold = metrics_df.loc[best_fold_idx]

    print(f"\n🔹 Model: {model_name}")
    print(f"   🏆 Best Fold: Fold {best_fold['Fold']}")
    print(f"   R2: {best_fold['R2']:.4f}")
    print(f"   RMSE: {best_fold['RMSE']:.4f}")
    print(f"   📈 Mean R2: {metrics_df['R2'].mean():.4f}")
    print(f"   📉 Mean RMSE: {metrics_df['RMSE'].mean():.4f}")
