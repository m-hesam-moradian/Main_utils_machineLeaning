import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your Excel file
file_path = r"D:\ML\Main_utils_machineLeaning\task\BSE. No.14-Dataset.xlsx1"
df = pd.read_excel(file_path)

# Create a copy to avoid modifying original
df_encoded = df.copy()

# Initialize encoder
encoder = LabelEncoder()

# Loop through columns and encode if dtype is object or category
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object" or df_encoded[col].dtype.name == "category":
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))

# Save to a new sheet in the same Excel file
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl") as writer:
    df_encoded.to_excel(writer, sheet_name="Encoded_Data", index=False)
