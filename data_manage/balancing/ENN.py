import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
import openpyxl

# --- Load data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Encoded_Data"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Apply Edited Nearest Neighbours (ENN) ---
enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_resample(X, y)

# --- Combine and shuffle rows ---
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced[target_column] = y_resampled
df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

# --- Save to Excel sheet ENN_Data ---
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="ENN_Data", index=False)

print("[+] Balanced data saved to sheet 'ENN_Data' in Data.xlsx using ENN.")
print("    Shape:", df_balanced.shape)
print("    Target distribution:\n", df_balanced[target_column].value_counts())
