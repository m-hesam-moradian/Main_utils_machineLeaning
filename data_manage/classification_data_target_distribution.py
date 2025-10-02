import pandas as pd

# --- Load dataset ---
sheet_name = "Data after K-FOLD"
excel_path = r"D:\ML\Main_utils\task\136_Seismic_ETC_RTHA, BO.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Target column ---
target_column = "Class"

# --- Count samples per class ---
class_counts = df[target_column].value_counts().reset_index()
class_counts.columns = ["ClassLabel", "SampleCount"]

# --- Save to variable ---
class_distribution_df = class_counts

# --- Display ---
print("\n📊 Sample count per class:")
print(class_distribution_df)
