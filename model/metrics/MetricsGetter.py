import pandas as pd
import numpy as np
from Metrics_regression import getAllMetric, REC

# --- File Paths ---
value_path = r"D:\ML\Main_utils\data\fakeValue.npt"
prediction_path = r"D:\ML\Main_utils\data\fakePrediction_updated.npt"

# --- Load Data ---
values_df = pd.read_csv(
    value_path, sep="\t", header=None, names=["value"], engine="python"
)
preds_df = pd.read_csv(
    prediction_path, sep="\t", header=None, names=["prediction"], engine="python"
)

# --- Merge and Convert ---
df = pd.concat([values_df, preds_df], axis=1)
values = df["value"].to_numpy()
predictions = df["prediction"].to_numpy()

# --- Split: Train (80%) / Test (20%) ---
split_idx = int(len(values) * 0.8)
y_train = values[:split_idx]
y_test = values[split_idx:]
y_pred_train = predictions[:split_idx]
y_pred_test = predictions[split_idx:]

# --- Split Test into Value / Test-Value ---
mid_idx = len(y_test) // 2
y_test_value = y_test[:mid_idx]
y_test_test_value = y_test[mid_idx:]
y_pred_value = y_pred_test[:mid_idx]
y_pred_test_value = y_pred_test[mid_idx:]

# --- Build Metrics Table ---
metrics_data = {"Set": [], "R2": [], "RMSE": [], "MAE": [], "RSE": [], "SMAPE": []}

sets = [
    ("All", values, predictions),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test_value, y_pred_value),
    ("Test-Value", y_test_test_value, y_pred_test_value),
]

for name, y_true, y_pred in sets:
    R2, RMSE, MAE, RSE, SMAPE = getAllMetric(y_true, y_pred)
    metrics_data["Set"].append(name)
    metrics_data["R2"].append(R2)
    metrics_data["RMSE"].append(RMSE)
    metrics_data["MAE"].append(MAE)
    metrics_data["RSE"].append(RSE)
    metrics_data["SMAPE"].append(SMAPE)

metrics_df = pd.DataFrame(metrics_data)

rec = REC(y_test, y_pred_test)
rec_df = pd.DataFrame(rec)
# --- Display ---
print("\n📊 Performance Metrics Table:")
print(metrics_df)
