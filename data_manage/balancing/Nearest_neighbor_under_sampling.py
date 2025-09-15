# D:\ML\Project-2\src\model\train_model.py

import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours

# Path & target
DATA_PATH = r"D:\ML\main_structure\data\data.xlsx"
TARGET = "HVAC Efficiency"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="data_after_k_fold")

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

print("Original dataset shape:", X.shape, y.shape)

# ===== Balance the entire dataset using Edited Nearest Neighbours (ENN) =====
enn = EditedNearestNeighbours()
X_bal, y_bal = enn.fit_resample(X, y)

print("Balanced dataset shape:", X_bal.shape, y_bal.shape)

# Concatenate features and target into a single DataFrame
dataafterbalancing = pd.concat(
    [pd.DataFrame(X_bal, columns=X.columns), pd.Series(y_bal, name=TARGET)], axis=1
)

print("\nBalanced DataFrame preview:")
print(dataafterbalancing.head())
