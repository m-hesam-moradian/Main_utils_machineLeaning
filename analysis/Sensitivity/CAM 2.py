import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from SALib.sample import saltelli
from SALib.analyze import sobol
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data
# ==========================================
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\Data.xlsx", sheet_name="Data_after_KFold_LR")
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
feature_names = list(X.columns)

y = pd.read_csv(r"C:\Users\Sam\Desktop\ML\data\predictions.txt", header=None).squeeze()

# Align lengths
min_len = min(X.shape[0], len(y))
X = X.iloc[:min_len]
y = y.iloc[:min_len]

# ==========================================
# 2. Fit Surrogate Model
# ==========================================
# Sobol requires sampling new combinations, so we fit the SVC to evaluate them
model = LinearSVC(dual=False, max_iter=10000)
model.fit(X, y)

# ==========================================
# 3. Setup SALib Problem & Sample
# ==========================================
bounds = [[X[col].min(), X[col].max()] for col in feature_names]
problem = {
    'num_vars': len(feature_names),
    'names': feature_names,
    'bounds': bounds
}

# Generate samples (required for S2 interactions)
param_values = saltelli.sample(problem, 1024, calc_second_order=True)
Y_pred = model.predict(param_values)

# ==========================================
# 4. Calculate S1, S2, ST Indices
# ==========================================
Si = sobol.analyze(problem, Y_pred, calc_second_order=True)

# Format S1 and ST into a DataFrame
df_s1_st = pd.DataFrame({
    "Parameter": problem['names'],
    "S1": Si['S1'],
    "ST": Si['ST']
})

# Format S2 into a DataFrame (Extracting the pairwise interactions)
s2_data = []
for i, name_i in enumerate(problem['names']):
    for j, name_j in enumerate(problem['names']):
        if i < j: # Only grab unique pairs
            s2_data.append({
                "Parameter_1": name_i,
                "Parameter_2": name_j,
                "S2": Si['S2'][i, j]
            })
df_s2 = pd.DataFrame(s2_data)

# ==========================================
# 5. Export and Print Results
# ==========================================
# Save files directly to your task folder
# Concatenate the two DataFrames into one (rows will stack)
df_combined = pd.concat([df_s1_st, df_s2], axis=0, ignore_index=True)

# Copy the entire combined table to your clipboard
df_combined.to_clipboard(index=False)

print("Combined S1, ST, and S2 results successfully copied to clipboard!")

print("--- First-Order (S1) and Total (ST) Indices ---")
print(df_s1_st.to_string(index=False))

print("\n--- Top 5 Second-Order (S2) Interactions ---")
print(df_s2.sort_values(by="S2", ascending=False).head().to_string(index=False))
print(f"\n[!] Full S2 table (45 rows) saved to: C:\\Users\\Sam\\Desktop\\ML\\task\\S2_Results.csv")