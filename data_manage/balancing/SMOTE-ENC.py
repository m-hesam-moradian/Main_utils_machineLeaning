import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# -------------------- 1. Load the data --------------------
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Encoded_Data_Test")

# Prepare Features (X) and Target (y)
target_column = df.columns[-1]
X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()

# -------------------- 2. Encode Target (if necessary) --------------------
le = LabelEncoder()

if y.dtype == 'object' or y.dtype.name == 'category':
    y_encoded = le.fit_transform(y)
else:
    y_encoded = y  # already numeric

# -------------------- 3. Oversampling (SMOTE) --------------------
smote = SMOTE(
    sampling_strategy='auto',  # balances all classes
    random_state=42,
    k_neighbors=5              # you can tune this
)

X_res, y_res_encoded = smote.fit_resample(X, y_encoded)

# -------------------- 4. Reconstruct DataFrame --------------------
df_balanced = pd.DataFrame(X_res, columns=X.columns)

if y.dtype == 'object' or y.dtype.name == 'category':
    df_balanced[target_column] = le.inverse_transform(y_res_encoded)
else:
    df_balanced[target_column] = y_res_encoded

# Shuffle dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution after SMOTE (oversampling):")
print(df_balanced[target_column].value_counts())

# -------------------- 5. Save to Excel --------------------
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="Balanced_SMOTE", index=False)