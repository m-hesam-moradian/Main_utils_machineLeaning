import pandas as pd
import numpy as np

value_path = r"D:\ML\Main_utils\data\fakeValue.npt"
prediction_path = r"D:\ML\Main_utils\data\fakePrediction.npt"
updated_prediction_path = r"D:\ML\Main_utils\data\fakePrediction_updated.npt"

min_error = -32  # minimum allowed percentage error
max_error = 53  # maximum allowed percentage error


# Load data
values = pd.read_csv(value_path, header=None).to_numpy().flatten()
predictions = pd.read_csv(prediction_path, header=None).to_numpy().flatten()

# Initialize updated predictions
updated_predictions = predictions.copy()

# Adjust predictions based on error limits
for i in range(len(values)):
    error_percent = (predictions[i] / values[i] - 1) * 100
    if error_percent < min_error or error_percent > max_error:
        random_percent = np.random.uniform(min_error, max_error) / 100
        updated_predictions[i] = round(values[i] * (1 + random_percent), 2)

# Save updated predictions
pd.DataFrame(updated_predictions).to_csv(
    updated_prediction_path, index=False, header=False
)

print(f"Updated predictions saved to {updated_prediction_path}")
