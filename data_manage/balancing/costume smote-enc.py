import pandas as pd
from smote_enc.smote_enc import SMOTEEncoder

# === CONFIGURATION ===
input_file = r'C:\Users\Sam\Desktop\ML\task\BSS.No.1-Dataset.xlsx'
input_sheet = 'BSS.No.1-Target 1 Z_Score'
output_sheet = 'BSS.No.1-Target 1 Balanced'

# === STEP 1: Load Cleaned Data ===
df = pd.read_excel(input_file, sheet_name=input_sheet)

# === STEP 2: Identify Target and Categorical Columns ===
target_col = 'Anomalous Load'
categorical_cols = df.select_dtypes(include='object').columns.tolist()

X = df.drop(columns=[target_col])
y = df[target_col]

# === STEP 3: Apply SMOTE-ENC ===
smote = SMOTEEncoder(categorical_features=categorical_cols, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

df_balanced = pd.concat([X_resampled, y_resampled], axis=1)

# === STEP 4: Save to New Sheet ===
with pd.ExcelWriter(input_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df_balanced.to_excel(writer, sheet_name=output_sheet, index=False)