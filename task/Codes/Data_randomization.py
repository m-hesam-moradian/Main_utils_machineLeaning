import pandas as pd

# --- Load original dataset ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
df = pd.read_excel(excel_path, sheet_name="CLEANED_DATA")

# --- Shuffle the dataset ---
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_clipboard(index=False)
# --- Save to same Excel file under new sheet ---
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    df_shuffled.to_excel(writer, sheet_name="DATA_Shuffled", index=False)

print("✅ Dataset randomized successfully.")
print(f"📁 Shuffled data saved to sheet 'DATA_Shuffled' in '{excel_path}'.")
