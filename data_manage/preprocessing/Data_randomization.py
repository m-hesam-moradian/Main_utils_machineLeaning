import pandas as pd

# --- Load original dataset ---
sheet_name = "CLEANED_DATA"
excel_path = r"D:\ML\Main_utils\task\Resource_utilization.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Shuffle the dataset ---
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Save to same Excel file under new sheet ---
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    df_shuffled.to_excel(writer, sheet_name="DATA_Shuffled", index=False)

print("✅ Dataset randomized successfully.")
print(f"📁 Shuffled data saved to sheet 'DATA_Shuffled' in '{excel_path}'.")
