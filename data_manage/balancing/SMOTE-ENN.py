import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

# -------------------- 1. Load the data --------------------
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Selected_Data_RFE")

# Prepare Features (X) and Target (y)
target_column = df.columns[-1]
X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()

# -------------------- 2. Encode Target (if necessary) --------------------
le = LabelEncoder()

if y.dtype == "object" or y.dtype.name == "category":
    y_encoded = le.fit_transform(y)
else:
    y_encoded = y

# -------------------- 3. Hybrid Balancing (SMOTE-ENN) --------------------
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

smote_enn = SMOTEENN(
    smote=SMOTE(random_state=42),
    enn=EditedNearestNeighbours(
        n_neighbors=3,
        kind_sel="mode"
    ),
    random_state=42
)

X_res, y_res_encoded = smote_enn.fit_resample(X, y_encoded)

# -------------------- 4. Reconstruct DataFrame --------------------
df_balanced = pd.DataFrame(X_res, columns=X.columns)

if y.dtype == "object" or y.dtype.name == "category":
    df_balanced[target_column] = le.inverse_transform(y_res_encoded)
else:
    df_balanced[target_column] = y_res_encoded

# Shuffle dataset
df_balanced = (
    df_balanced
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

print("Original class distribution:")
print(pd.Series(y).value_counts())

print("\nClass distribution after SMOTE-ENN:")
print(df_balanced[target_column].value_counts())

print(f"\nOriginal dataset size : {len(df)}")
print(f"Balanced dataset size : {len(df_balanced)}")

# -------------------- 5. Save to Excel --------------------
with pd.ExcelWriter(
    file_path,
    mode="a",
    engine="openpyxl",
    if_sheet_exists="replace"
) as writer:
    df_balanced.to_excel(
        writer,
        sheet_name="Balanced_SMOTE_ENN",
        index=False
    )

print("\nBalanced dataset saved successfully to sheet 'Balanced_SMOTE_ENN'.")