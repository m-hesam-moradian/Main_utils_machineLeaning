import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- Load dataset ---
sheet_name = "DATA_Shuffled"
excel_path = r"D:\ML\Main_utils\task\EL. No 6. Allocated bandwidth- SVR-ENR-SCO-POA-GGO-DATA.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Target column ---
target_column = "allocated_bandwidth"

# --- Separate features and target ---
features = df.drop(columns=[target_column])
target = df[[target_column]]  # Keep as DataFrame for clean concat

# --- Normalize numeric features only ---
numeric_cols = features.select_dtypes(include="number").columns
scaler = MinMaxScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

# --- Recombine normalized features with untouched target ---
df_normalized = pd.concat([features, target], axis=1)

# --- Save to same Excel file under new sheet ---
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    df_normalized.to_excel(writer, sheet_name="DATA_Normalized", index=False)

print(
    f"✅ Normalized features saved to sheet 'DATA_Normalized' in '{excel_path}'. Target column untouched."
)
