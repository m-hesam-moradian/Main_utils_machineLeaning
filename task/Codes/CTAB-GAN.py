import pandas as pd
from ctgan import CTGAN

# Load data
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="Scenario 1 Data")
target = 'Anomalous Load'

# Focus only on relevant columns
features = ['Latency', 'Bandwidth Utilization', target]
df_subset = df[features]

# Identify imbalance
counts = df_subset[target].value_counts()
minority_class = counts.idxmin()
majority_count = counts.max()
minority_count = counts.min()
need = majority_count - minority_count

print(f"Original: {counts.to_dict()}")
print(f"Need {need} samples of class {minority_class}")

# Train CTGAN
model = CTGAN(epochs=50, verbose=True)
model.fit(df_subset, discrete_columns=['Latency', 'Bandwidth Utilization', target])
synthetic = model.sample(need * 2)


# Filter for minority class
synthetic_minority = synthetic[synthetic[target] == minority_class]
while len(synthetic_minority) < need:
    more = model.sample(need * 2)
    more_min = more[more[target] == minority_class]
    synthetic_minority = pd.concat([synthetic_minority, more_min], ignore_index=True)

synthetic_minority = synthetic_minority.head(need)

# Combine with original
balanced_df = pd.concat([df_subset, synthetic_minority], ignore_index=True)

# Copy to 
balanced_df.to_clipboard(index=False)
print("\nBALANCED DATA COPIED TO CLIPBOARD!")
print(balanced_df[target].value_counts())
