import pandas as pd
from ctgan import CTGAN

# -------------------------------
# 1. Load your Excel
# -------------------------------
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="BSS.No.1-Target 1")

# -------------------------------
# 2. Settings
# -------------------------------
target = 'Anomalous Load'
categorical_cols = ['Protocol Type']  # Only this is categorical
discrete_columns = categorical_cols + [target]

# Count classes
counts = df[target].value_counts()
minority_class = counts.idxmin()
majority_count = counts.max()
minority_count = counts.min()
need = majority_count - minority_count

print(f"Original: {counts.to_dict()}")
print(f"Need {need} samples of class {minority_class}")

# -------------------------------
# 3. Train CTGAN
# -------------------------------
model = CTGAN(epochs=300, verbose=True)
model.fit(df, discrete_columns=discrete_columns)

# -------------------------------
# 4. Generate 3x needed, then filter minority
# -------------------------------
extra = need * 3  # Generate more to ensure we get enough
synthetic = model.sample(extra)

# Keep only minority class
synthetic_minority = synthetic[synthetic[target] == minority_class]

# If not enough, generate more
while len(synthetic_minority) < need:
    more = model.sample(need * 2)
    more_min = more[more[target] == minority_class]
    synthetic_minority = pd.concat([synthetic_minority, more_min], ignore_index=True)

# Take exactly what we need
synthetic_minority = synthetic_minority.head(need)

# -------------------------------
# 5. Combine → BALANCED
# -------------------------------
balanced_df = pd.concat([df, synthetic_minority], ignore_index=True)

# -------------------------------
# 6. Done! Copy to clipboard
# -------------------------------
balanced_df.to_clipboard(index=False)
print("\nBALANCED DATA COPIED TO CLIPBOARD!")
print(balanced_df[target].value_counts())