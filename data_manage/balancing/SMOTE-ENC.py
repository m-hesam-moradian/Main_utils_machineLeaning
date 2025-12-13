import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN

# -------------------- 1. Load the data --------------------
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Encoded_Data")

target_col = "Fault Label"  # or df.columns[-1] if last column is target
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Encode categorical features if needed
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])

# -------------------- 2. Apply SMOTE-ENN with fixed class size --------------------
# Define target sample size per class
sampling_strategy = {cls: 1150 for cls in np.unique(y)}

smoteenn = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
X_balanced, y_balanced = smoteenn.fit_resample(X_encoded, y)

# -------------------- 3. Final result --------------------
df_balanced = pd.DataFrame(X_balanced, columns=X_encoded.columns)
df_balanced[target_col] = y_balanced
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution after balancing:")
print(df_balanced[target_col].value_counts())

# -------------------- 4. Save to Excel --------------------
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="Balanced_SMOTEENN", index=False)