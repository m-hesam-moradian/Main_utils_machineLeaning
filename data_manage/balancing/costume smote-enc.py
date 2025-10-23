import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC


# === Step 1: Load original Excel file ===
file_path = r"C:\Users\Sam\Desktop\ML\task\BSS.No.1-Dataset.xlsx"
sheet_name = "BSS.No.1-Target 1"


df = pd.read_excel(file_path, sheet_name=sheet_name)

# === Step 2: Separate features and target ===
target_col = "Anomalous Load"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in data.")

X = df.drop(columns=[target_col])
y = df[target_col]

# === Step 3: Encode categorical features ===
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
cat_indices = [X.columns.get_loc(col) for col in categorical_features]  # ✅ FIXED: use indices

encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# === Step 4: Apply SMOTENC ===
if not cat_indices:
    raise ValueError("SMOTENC requires at least one categorical feature. None found.")

smote = SMOTENC(categorical_features=cat_indices, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# === Step 5: Reconstruct balanced DataFrame ===
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced[target_col] = y_resampled

# === Step 6: Save to new sheet in same Excel file ===
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_balanced.to_excel(writer, sheet_name="Balanced_SMOTENC", index=False)

print("✅ Balanced dataset saved to 'Balanced_SMOTENC' sheet.")