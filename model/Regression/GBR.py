# D:\ML\Project-2\src\model\gbr_regression_metrics.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from Metrics_regression import getAllMetric

# --- Path & target ---
DATA_PATH = (
    r"D:\ML\Main_utils\Task\Original Dataset- Concrete (elevated temperature).xlsx"
)
SHEET_NAME = "Standard_Normalized"
TARGET = "Compressive Strength"

# --- Load dataset ---
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME).dropna()

# --- Features & target ---
X = df.drop(columns=[TARGET])
y = df[TARGET]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Define GBR Model ---
model = GradientBoostingRegressor(
    n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
)

# --- Train model ---
model.fit(X_train, y_train)

# --- Predictions ---
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# --- Split test into halves ---
half = len(X_test) // 2
y_test_first_half, y_test_first_pred = y_test.iloc[:half], y_test_pred[:half]
y_test_second_half, y_test_second_pred = y_test.iloc[half:], y_test_pred[half:]

# --- Collect metrics using your custom function ---
all_metrics = getAllMetric(y, model.predict(X))
train_metrics = getAllMetric(y_train, y_train_pred)
test_metrics = getAllMetric(y_test, y_test_pred)
value_metrics = getAllMetric(y_test_first_half, y_test_first_pred)
test_half_metrics = getAllMetric(y_test_second_half, y_test_second_pred)

metrics_df = pd.DataFrame(
    [
        {
            "Dataset": "All",
            "R": all_metrics[0],
            "RMSE": all_metrics[1],
            "MAE": all_metrics[2],
            "RSE": all_metrics[3],
            "SMAPE": all_metrics[4],
        },
        {
            "Dataset": "Train",
            "R": train_metrics[0],
            "RMSE": train_metrics[1],
            "MAE": train_metrics[2],
            "RSE": train_metrics[3],
            "SMAPE": train_metrics[4],
        },
        {
            "Dataset": "Test",
            "R": test_metrics[0],
            "RMSE": test_metrics[1],
            "MAE": test_metrics[2],
            "RSE": test_metrics[3],
            "SMAPE": test_metrics[4],
        },
        {
            "Dataset": "Value",
            "R": value_metrics[0],
            "RMSE": value_metrics[1],
            "MAE": value_metrics[2],
            "RSE": value_metrics[3],
            "SMAPE": value_metrics[4],
        },
        {
            "Dataset": "Test_Half",
            "R": test_half_metrics[0],
            "RMSE": test_half_metrics[1],
            "MAE": test_half_metrics[2],
            "RSE": test_half_metrics[3],
            "SMAPE": test_half_metrics[4],
        },
    ]
)

# --- Show results ---
print("\n--- Performance Metrics (Gradient Boosting Regressor) ---")
print(metrics_df)
