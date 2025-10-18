import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.preprocessing import LabelEncoder

# Load Excel file
file_path = "BSE_No14.xlsx"  # Update this if your file name is different
df = pd.read_excel(file_path)

# Separate features and target
target_column = "Anomaly Detected"
X = df.drop(columns=[target_column])
y = df[target_column]

# Encode target if it's not numeric
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# Apply ENN for under-sampling
enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_resample(X, y)

# Combine resampled data into a new DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled[target_column] = y_resampled

# Save the balanced dataset
df_resampled.to_excel("BSE_No14_ENN_balanced.xlsx", index=False)
print("Balanced dataset saved as 'BSE_No14_ENN_balanced.xlsx'")
