import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your Excel file
df = pd.read_excel(
    r"D:\ML\Main_utils\task\WA_Fn-UseC_-HR-Employee-Attrition.xlsx"
)  # Replace with your actual filename

# Create a copy to avoid modifying original
df = df.dropna()
df_encoded = df.copy()

# Initialize encoder
encoder = LabelEncoder()

# Loop through columns and encode if dtype is object or category
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object" or df_encoded[col].dtype.name == "category":
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))

# Display the encoded DataFrame
print(df_encoded.head())
