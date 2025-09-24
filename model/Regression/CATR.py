import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from Metrics_regression import getAllMetric
import numpy as np

sheet_name = "Data after K-Fold (GBR & ANFIS)"
df = pd.read_excel(
    r"D:\ML\Main_utils\Task\Global_AI_Content_Impact_Dataset.xlsx",
    sheet_name=sheet_name,
)
target_column = "Market Share of AI Companies (%)"

# --- Features and Target ---
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# --- CatBoost Regressor Model with Default Parameters ---
model = CatBoostRegressor(
    iterations=1000,  # Default: 1000
    learning_rate=None,  # Default: None (auto-selected based on dataset size)
    depth=6,  # Default: 6
    l2_leaf_reg=3.0,  # Default: 3.0
    loss_function="RMSE",  # Default: 'RMSE' for regression
    random_seed=42,  # Default: None, set to 42 for reproducibility
    verbose=0,  # Default: 0 (silent mode, no training output)
)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Split Test Predictions ---
mid_index = len(y_pred_test) // 2
y_test_first_half = y_test.iloc[:mid_index]
y_test_second_half = y_test.iloc[mid_index:]
y_pred_test_first_half = y_pred_test[:mid_index]
y_pred_test_second_half = y_pred_test[mid_index:]

# --- Build Metrics Table Using getAllMetric ---
metrics_data = {"Set": [], "R2": [], "RMSE": [], "MAE": [], "RSE": [], "SMAPE": []}
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test_first_half, y_pred_test_first_half),
    ("Test-Value", y_test_second_half, y_pred_test_second_half),
]

for name, y_true, y_pred in sets:
    R, RMSE, MAE, RSE, SMAPE = getAllMetric(y_true, y_pred)
    metrics_data["Set"].append(name)
    metrics_data["R2"].append(R)
    metrics_data["RMSE"].append(RMSE)
    metrics_data["MAE"].append(MAE)
    metrics_data["RSE"].append(RSE)
    metrics_data["SMAPE"].append(SMAPE)

metrics_df = pd.DataFrame(metrics_data)

df_train = pd.DataFrame({"y_train_real": y_train.values, "y_train_pred": y_pred_train})
df_test = pd.DataFrame({"y_test_real": y_test.values, "y_test_pred": y_pred_test})
df_all = pd.concat(
    [
        pd.DataFrame({"y_real": y_train.values, "y_pred": y_pred_train}),
        pd.DataFrame({"y_real": y_test.values, "y_pred": y_pred_test}),
    ],
    ignore_index=True,
)

print("\n📋 Performance Metrics Table:")
print(metrics_df)
