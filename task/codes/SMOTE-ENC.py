import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

# -------------------- 1. Load the data --------------------
file_path = r"D:\ML\ML\task\BSE. No.13-Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="DAtA after VIF")

# Display initial info
print("Columns:", df.columns)
print(df.head())
print(df.dtypes)

# -------------------- 2. Identify features and target --------------------
# Assumption: last column is the target
target_col = df.columns[-1]  # or use 'target' if you know the exact name
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

# -------------------- 3. Encode categorical columns --------------------
X_encoded = X.copy()
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    le_dict[col] = le  # Save encoder for future use or decoding

# -------------------- 4. Oversampling with SMOTE or SMOTENC --------------------
if len(categorical_cols) > 0:
    # Use SMOTENC for mixed data
    categorical_features_idx = [X_encoded.columns.get_loc(c) for c in categorical_cols]
    smote = SMOTENC(categorical_features=categorical_features_idx, random_state=42)
    print("Using SMOTENC for mixed (numeric + categorical) data")
else:
    # Use regular SMOTE for numeric-only data
    smote = SMOTE(random_state=42)
    print("Using regular SMOTE for numeric data")

X_over, y_over = smote.fit_resample(X_encoded, y)
print("After Oversampling:", X_over.shape, y_over.shape)

# -------------------- 5. Undersampling with ENN --------------------
enn = EditedNearestNeighbours()
X_balanced, y_balanced = enn.fit_resample(X_over, y_over)
print("After ENN:", X_balanced.shape, y_balanced.shape)


# -------------------- 6. Final result --------------------
# Convert back to DataFrame
df_balanced = pd.DataFrame(X_balanced, columns=X_encoded.columns)
df_balanced[target_col] = y_balanced

# Shuffle the rows randomly
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Sample of final balanced data:")
print(df_balanced.head())
print("Class distribution after balancing:")
print(df_balanced[target_col].value_counts())

# -------------------- 7. Save to a new sheet in the same Excel file --------------------
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl") as writer:
    df_balanced.to_excel(writer, sheet_name="Balanced_Shuffled", index=False)
