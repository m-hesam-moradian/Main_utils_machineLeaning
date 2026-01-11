import pandas as pd
import numpy as np

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

# Read data from Excel
df = pd.read_excel(excel_path, sheet_name="Encoded_Data")
normalized_df = pd.read_excel(excel_path, sheet_name="DATA_Normalized")

# Calculate average, min, and max for each feature and target
averages = df.mean()
mins = df.min()
maxs = df.max()

# Add white Gaussian noise to original data
np.random.seed(0)
noisy_df = df.copy()
for col in df.columns:
    mean = averages[col]
    std_dev = (maxs[col] - mins[col]) / 6  # Using 6-sigma rule
    noise = np.random.normal(0, std_dev, size=len(df))  # Changed mean to 0
    noisy_df[col] = df[col] + noise


# Add white Gaussian noise to normalized data
normalized_noisy_df = normalized_df.copy()
for col in normalized_df.columns:
    mean = 0  # Normalized data has mean 0
    std_dev = 0.1  # Assuming a small standard deviation for normalized data
    noise = np.random.normal(mean, std_dev, size=len(normalized_df))
    normalized_noisy_df[col] = normalized_df[col] + noise

# Save noisy normalized data to a new Excel file
normalized_noisy_df.to_clipboard(index=False)