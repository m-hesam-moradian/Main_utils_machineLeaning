import pandas as pd

# --- Load dataset ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data"
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Target column ---
target_column = df.columns[-1]

# --- Class distribution ---
class_counts = df[target_column].value_counts()
total_samples = len(df)
class_percentages = class_counts / total_samples * 100

print("\n📊 Sample count per class:")
print(class_counts)

print("\n📊 Class percentages:")
print(class_percentages.round(2))

# --- Check for imbalance ---
max_class_pct = class_percentages.max()
if max_class_pct > 80:
    print(f"\n🚨 Class imbalance detected: One class = {max_class_pct:.2f}% of total samples.")
else:
    print("\n✅ Class distribution appears balanced.")