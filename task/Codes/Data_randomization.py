import pandas as pd

# --- Load original dataset ---
sheet_name = "DATA"
excel_path = r"D:\ML\Main_utils\task\EI_No_3__Optimal Scheduling_Classification_DTC_RFR_XGBC_HOA_DOA_Data.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Shuffle the dataset ---
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("✅ Dataset randomized successfully.")

# --- Save to same Excel file under new sheet ---
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    df_shuffled.to_excel(writer, sheet_name="DATA_Shuffled", index=False)

print(f"📁 Shuffled data saved to sheet 'DATA_Shuffled' in '{excel_path}'.")
