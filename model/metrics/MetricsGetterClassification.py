import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from getAllMetrics_Classification import getAllMetric

# Load data
df = np.loadtxt(r"D:\ML\Main_utils\data\Data_err.npt")
values = df[:, 0]  # True values
predictions = df[:, 1]  # Predicted values

# --- Binarize the Target Variable ---
# Convert continuous values to binary (0/1) using median as threshold
median_value = np.median(values)
y = (values > median_value).astype(int)  # True binary labels
y_pred = (predictions > median_value).astype(int)  # Predicted binary labels

# --- Split: Train (80%) / Test (20%) ---
split_idx = int(len(y) * 0.8)
y_train = y[:split_idx]
y_test = y[split_idx:]
y_pred_train = y_pred[:split_idx]
y_pred_test = y_pred[split_idx:]

# --- Split Test into Value / Test-Value ---
mid_idx = len(y_test) // 2
y_test_value = y_test[:mid_idx]
y_test_test_value = y_test[mid_idx:]
y_pred_value = y_pred_test[:mid_idx]
y_pred_test_value = y_pred_test[mid_idx:]

# --- Build Metrics Table ---
metrics_data = {
    "Set": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "F2": [],
}

sets = [
    ("All", y, y_pred),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test_value, y_pred_value),
    ("Test-Value", y_test_test_value, y_pred_test_value),
]

for name, y_true, y_pred in sets:
    Accuracy, Precision, Recall, F1, F2 = getAllMetric(y_true, y_pred)
    metrics_data["Set"].append(name)
    metrics_data["Accuracy"].append(Accuracy)
    metrics_data["Precision"].append(Precision)
    metrics_data["Recall"].append(Recall)
    metrics_data["F1"].append(F1)
    metrics_data["F2"].append(F2)

metrics_df = pd.DataFrame(metrics_data)

# --- Create DataFrames for real vs predicted ---
df_train = pd.DataFrame({"y_train_real": y_train, "y_train_pred": y_pred_train})
df_test = pd.DataFrame({"y_test_real": y_test, "y_test_pred": y_pred_test})
df_all = pd.concat(
    [
        pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train}),
        pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test}),
    ],
    ignore_index=True,
)

# --- Display Metrics Table ---
print("\n📋 Performance Metrics Table:")
print(metrics_df)
