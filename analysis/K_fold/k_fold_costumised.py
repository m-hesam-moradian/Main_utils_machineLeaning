from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np

# Load data
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="DATA_Shuffled")
X = df.drop(columns=["Anomalous Load"])
y = df["Anomalous Load"]

# Initialize model and KFold
model = CatBoostRegressor(verbose=0)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics_list = []
fold_data_list = []
fold_indices = []

# First pass: train and collect metrics + indices
for fold_index, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    fold_metrics_list.append({"Fold": fold_index, "R2": r2, "RMSE": rmse})
    fold_data_list.append(df.iloc[test_idx])
    fold_indices.append(test_idx)

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(fold_metrics_list)

# Identify best fold by highest R²
best_fold_index = metrics_df["R2"].idxmax()
best_fold_data = fold_data_list[best_fold_index]

# Reorder: all other folds first, best fold last
non_best_folds = [fold_data_list[i] for i in range(len(fold_data_list)) if i != best_fold_index]
reordered_data = pd.concat(non_best_folds + [best_fold_data], ignore_index=True)

# Save to variable
final_dataset = reordered_data

# Output
print("Fold Metrics Table:")
print(metrics_df)
print("\nBest fold:", metrics_df.iloc[best_fold_index]["Fold"])
print("Final dataset shape:", final_dataset.shape)