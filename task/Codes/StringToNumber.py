import pandas as pd
from sklearn.preprocessing import LabelEncoder
from openpyxl import load_workbook

# Load your Excel file
excel_path = r"D:\ML\Main_utils\task\EL. No 6. Allocated bandwidth- SVR-ENR-SCO-POA-GGO-DATA.xlsx"
df = pd.read_excel(excel_path, sheet_name="DATA_Normalized")

# Create a copy to avoid modifying original
df_encoded = df.copy()

# Initialize encoder
encoder = LabelEncoder()

# Encode only columns where all non-null values are strings
for col in df_encoded.columns:
    non_null_values = df_encoded[col].dropna()
    if df_encoded[col].dtype == "object" or df_encoded[col].dtype.name == "category":
        if non_null_values.map(type).eq(str).all():
            encoded = encoder.fit_transform(non_null_values)
            df_encoded.loc[non_null_values.index, col] = encoded

# Save to a new sheet in the same Excel file
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="new"
) as writer:
    df_encoded.to_excel(writer, sheet_name="String labelEncoded", index=False)

print("✅ Encoded string columns and saved to sheet 'String labelEncoded'.")
