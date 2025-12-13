import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# -------------------- 1. Load the data --------------------
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Encoded_Data")

target_col = "Fault Label"
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Encode categorical features if needed
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])

# -------------------- 2. Oversample with SMOTE to 1150 per class --------------------
sampling_strategy = {"0": 1150, "1": 1150, "2": 1150}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_over, y_over = smote.fit_resample(X_encoded, y)

# -------------------- 3. Final result --------------------
df_balanced = pd.DataFrame(X_over, columns=X_encoded.columns)
df_balanced[target_col] = y_over
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution after balancing:")
print(df_balanced[target_col].value_counts())

# -------------------- 4. Save to Excel --------------------
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="Balanced_SMOTE", index=False)