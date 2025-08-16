import pandas as pd
from getAllMetric import getAllMetric


# File paths
# value_path = "fakeValue.npt"
# prediction_path = "fakePrediction.npt"


def getMetrics(y, updated_predictions):
    # Convert NumPy arrays to DataFrames
    y_df = pd.DataFrame(y, columns=["value"])
    preds_df = pd.DataFrame(updated_predictions, columns=["prediction"])

    # Merge into one DataFrame
    df = pd.concat([y_df, preds_df], axis=1)

    # Convert to NumPy arrays
    values = df["value"].to_numpy()
    predictions = df["prediction"].to_numpy()

    # Define split index for train/test (80% train, 20% test)
    split_idx = int(len(values) * 0.8)

    # Train and test splits
    train_values = values[:split_idx]
    train_preds = predictions[:split_idx]

    test_values = values[split_idx:]
    test_preds = predictions[split_idx:]

    # Split test into two halves
    mid_test_idx = split_idx + len(test_values) // 2

    test_first_half_values = values[split_idx:mid_test_idx]
    test_first_half_preds = predictions[split_idx:mid_test_idx]

    test_second_half_values = values[mid_test_idx:]
    test_second_half_preds = predictions[mid_test_idx:]

    # Collect metrics for each set
    metrics = pd.DataFrame(
        [
            getAllMetric(values, predictions),  # 1. All
            getAllMetric(train_values, train_preds),  # 2. Train
            getAllMetric(test_values, test_preds),  # 3. Test
            getAllMetric(
                test_first_half_values, test_first_half_preds
            ),  # 4. First half
            getAllMetric(
                test_second_half_values, test_second_half_preds
            ),  # 5. Second half
        ],
        index=["All", "Train", "Test", "value", "Test"],
    )
    train_metric = getAllMetric(train_values, train_preds)
    train_RMSE = train_metric["RMSE"]
    return metrics, train_RMSE


# metrics = getMetrics()
