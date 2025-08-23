import pandas as pd
import numpy as np
from getAllMetric import getAllMetric

# Example: load your values and predictions
value_path = "fakeValue.npt"
prediction_path = "fakePrediction_updated.npt"

values_df = pd.read_csv(
    value_path, sep="\t", header=None, names=["value"], engine="python"
)
preds_df = pd.read_csv(
    prediction_path, sep="\t", header=None, names=["prediction"], engine="python"
)

df = pd.concat([values_df, preds_df], axis=1)

# Convert to NumPy arrays
y = df["value"].to_numpy()
predictions = df["prediction"].to_numpy()

# Replace zeros with small random numbers to prevent divide by zero
y[y == 0] = np.random.uniform(0.01, 0.1, size=np.sum(y == 0))
predictions[predictions == 0] = np.random.uniform(
    0.01, 0.1, size=np.sum(predictions == 0)
)

# Split into train (80%) and test (20%)
split_idx = int(len(y) * 0.8)
y_train = y[:split_idx]
y_test = y[split_idx:]
pred_train = predictions[:split_idx]
pred_test = predictions[split_idx:]

# Split test into two halves: value and test_value
half_idx = len(y_test) // 2
y_test_first_half = y_test[:half_idx]
y_test_second_half = y_test[half_idx:]
y_test_first_pred = pred_test[:half_idx]
y_test_second_pred = pred_test[half_idx:]

# Compute metrics
metrics_dict = {
    "All": getAllMetric(y, predictions),
    "Train": getAllMetric(y_train, pred_train),
    "Test": getAllMetric(y_test, pred_test),
    "Value": getAllMetric(y_test_first_half, y_test_first_pred),
    "Test_value": getAllMetric(y_test_second_half, y_test_second_pred),
}

metrics_df = (
    pd.DataFrame(metrics_dict).T.reset_index().rename(columns={"index": "Metrics"})
)

print(metrics_df)


# # Define split index for train/test (80% train, 20% test)
# split_idx = int(len(values) * 0.8)

# # Train and test splits
# train_values = values[:split_idx]
# train_preds = predictions[:split_idx]

# test_values = values[split_idx:]
# test_preds = predictions[split_idx:]

# # Split test into two halves
# mid_test_idx = split_idx + len(test_values) // 2

# test_first_half_values = values[split_idx:mid_test_idx]
# test_first_half_preds = predictions[split_idx:mid_test_idx]

# test_second_half_values = values[mid_test_idx:]
# test_second_half_preds = predictions[mid_test_idx:]

# Collect metrics for each set

# metrics = pd.DataFrame(
#     [
#         getAllMetric(values, predictions),  # 1. All
#         getAllMetric(train_values, train_preds),  # 2. Train
#         getAllMetric(test_values, test_preds),  # 3. Test
#         getAllMetric(
#             test_first_half_values, test_first_half_preds
#         ),  # 4. First half of test
#         getAllMetric(
#             test_second_half_values, test_second_half_preds
#         ),  # 5. Second half of test
#     ],
#     index=["All", "Train", "Test", "Test_First_Half", "Test_Second_Half"],
# )

# print(metrics)
