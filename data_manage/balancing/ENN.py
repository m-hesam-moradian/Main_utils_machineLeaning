import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours

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

# --- Combine and copy to clipboard ---
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced[target_column] = y_resampled

df_balanced.to_clipboard(index=False)
print("✅ Balanced data copied to clipboard using ENN.")
