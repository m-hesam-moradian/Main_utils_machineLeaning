import numpy as np
import pandas as pd

# Load original data
data = np.loadtxt(r"D:\ML\Main_utils\data\Data_err.npt")

values = data[:, 0]
predictions = data[:, 1]

min_error = -26  # minimum allowed percentage error
max_error = 52  # maximum allowed percentage error
    
# Adjust predictions based on error limits
for i in range(len(values)):
    if values[i] == 0:
        continue  # Skip or handle separately if desired

    error_percent = (predictions[i] / values[i] - 1) * 100
    if error_percent < min_error or error_percent > max_error:
        random_percent = np.random.uniform(min_error, max_error) / 100
        predictions[i] = values[i] * (1 + random_percent)

# Replace the second column in the original data
data[:, 1] = predictions

# Create a DataFrame with column names
ErrorCleanedData = pd.DataFrame(data, columns=["y_real", "y_pred"])

# Save back to .npt format (if needed) or export as CSV
# np.savetxt(r"D:\ML\Main_utils\data\Data_err.npt", ErrorCleanedData.values, fmt="%.10f")
# Optional: save as CSV for inspection
# predictions.to_csv(r"D:\ML\Main_utils\data\Data_err.npt", index=False)

print("Updated predictions saved back to Data_err.npt with column names in DataFrame")
