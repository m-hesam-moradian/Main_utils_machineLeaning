import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.preprocessing import LabelEncoder

# Load Excel file
file_path = r"D:\ML\Main_utils_machineLeaning\task\BSE. No.14-Dataset.xlsx"  # Update this if your file name is different
df = pd.read_excel(file_path, sheet_name="Encoded_Data")

# Separate features and target
target_column = "Anomaly_Detected"
X = df.drop(columns=[target_column])
y = df[target_column]

# Apply ENN for under-sampling
enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_resample(X, y)

# Combine resampled data into a new DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled[target_column] = y_resampled


with pd.ExcelWriter(
    file_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    df_resampled.to_excel(writer, sheet_name="ENN_Balanced_Data", index=False)
print("✅ Dataset ENN_Balanced successfully.")

# --- Shuffle the dataset ---
df_shuffled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Save to same Excel file under new sheet ---
with pd.ExcelWriter(
    file_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    df_shuffled.to_excel(writer, sheet_name="DATA_Shuffled", index=False)

print("✅ Dataset randomized successfully.")
