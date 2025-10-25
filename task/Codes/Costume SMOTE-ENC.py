import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

# -------------------- 1. Load the data --------------------
file_path = r"C:\Users\Sam\Desktop\ML\task\BSS.No.1-Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="BSS.No.1-Target 1")

target_col = df.columns[-1]
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# -------------------- 2. Identify categorical and numeric columns --------------------
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# -------------------- 3. Encode categorical columns using SMOTE-ENC logic --------------------
X_encoded = X.copy()
imbalance_ratio = y.value_counts().min() / len(y)

for col in categorical_cols:
    label_stats = {}
    total_label_counts = X[col].value_counts()
    minority_label_counts = X[y == y.value_counts().idxmin()][col].value_counts()
    
    for label in total_label_counts.index:
        e = total_label_counts[label]
        o = minority_label_counts.get(label, 0)
        expected_e = e * imbalance_ratio
        chi = (o - expected_e)
        label_stats[label] = chi
    
    # Normalize and scale
    chi_values = np.array(list(label_stats.values()))
    if len(numeric_cols) > 0:
        std_median = np.median(X[numeric_cols].std())
        label_stats = {k: v * std_median for k, v in label_stats.items()}
    
    # Replace labels with encoded chi values
    X_encoded[col] = X[col].map(label_stats)

# -------------------- 4. Oversampling with SMOTE --------------------
smote = SMOTE(random_state=42)
X_over, y_over = smote.fit_resample(X_encoded, y)

# -------------------- 5. Undersampling with ENN --------------------
enn = EditedNearestNeighbours()
X_balanced, y_balanced = enn.fit_resample(X_over, y_over)

# -------------------- 6. Final result --------------------
df_balanced = pd.DataFrame(X_balanced, columns=X_encoded.columns)
df_balanced[target_col] = y_balanced
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------- 7. Save to Excel --------------------
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl") as writer:
    df_balanced.to_excel(writer, sheet_name="Balanced_SMOTEENC", index=False)