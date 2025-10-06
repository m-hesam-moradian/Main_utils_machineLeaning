import pandas as pd

# --- Load dataset ---
sheet_name = "DATA"
excel_path = r"D:\ML\Main_utils\task\EI_No_3__Optimal Scheduling_Classification_DTC_RFR_XGBC_HOA_DOA_Data.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Target column ---
target_column = "Target"

# --- Count samples per class ---
class_counts = df[target_column].value_counts().reset_index()
class_counts.columns = ["ClassLabel", "SampleCount"]

# --- Save to variable ---
class_distribution_df = class_counts

# --- Display ---
print("\n📊 Sample count per class:")
print(class_distribution_df)
