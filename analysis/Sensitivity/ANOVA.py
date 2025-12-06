from scipy.stats import f_oneway
import numpy as np
import pandas as pd

DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="Selected_Data")
target_column = df.columns[-1]

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

def anova_features_dataset(x, y, feature_names):
    results = []
    x = np.array(x)
    for i in range(x.shape[1]):
        f, p = f_oneway(x[:, i], y)   # use i, not i-1
        results.append({"Feature": feature_names[i], "F-statistic": f, "p-value": p})
    return pd.DataFrame(results)

# Run ANOVA
ANOVA = anova_features_dataset(X, y, X.columns)

# Copy to clipboard
ANOVA.to_clipboard(index=False)

# Optional: preview in console
print("✅ ANOVA results copied to clipboard")
print(ANOVA.head())