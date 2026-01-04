import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN

# -------------------- 1. Load the data --------------------
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Timestamp Removed")

# Prepare Features (X) and Target (y)
target_column = df.columns[-1]
X = df.drop(columns=[target_column]).copy() # Features are numeric
y = df[target_column].copy()

# -------------------- 2. Encode Target (if necessary) --------------------
# SMOTEENN requires classes to be numeric.
# If your target is text (e.g., "Normal", "Fault"), this converts it to 0, 1.
# If your target is already numeric (0, 1, 2), this does nothing.
le = LabelEncoder()
if y.dtype == 'object' or y.dtype.name == 'category':
    y_encoded = le.fit_transform(y)
else:
    y_encoded = y # Already numeric

# -------------------- 3. Hybrid Balancing (SMOTEENN) --------------------
# Automatically balances classes by oversampling minority and cleaning noise.
smote_enn = SMOTEENN(random_state=42)
X_res, y_res_encoded = smote_enn.fit_resample(X, y_encoded)

# -------------------- 4. Reconstruct DataFrame --------------------
# 1. Put features back
df_balanced = pd.DataFrame(X_res, columns=X.columns)

# 2. Inverse transform target back to original labels (if they were text)
if y.dtype == 'object' or y.dtype.name == 'category':
    df_balanced[target_column] = le.inverse_transform(y_res_encoded)
else:
    df_balanced[target_column] = y_res_encoded

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution after SMOTEENN balancing:")
print(df_balanced[target_column].value_counts())

# -------------------- 5. Save to Excel --------------------
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="Balanced_SMOTEENN", index=False)