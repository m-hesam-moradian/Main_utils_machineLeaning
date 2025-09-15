# D:\ML\Project-2\src\model\normality_check.py

import pandas as pd
from scipy.stats import shapiro

# Path & target
DATA_PATH = r"D:\ML\main_structure\data\data.xlsx"
TARGET = "HVAC Efficiency"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="data")

# Separate features and target (assuming train dataset)
X_train = df.drop(columns=[TARGET])

# Initialize a list to store results
normality_results = []

# Shapiro-Wilk test for each variable
for col in X_train.columns:
    stat, p_value = shapiro(X_train[col])
    normality_results.append(
        {
            "Variable": col,
            "Shapiro-Wilk Statistic": stat,
            "p-value": p_value,
            "Normal": p_value > 0.05,  # True if p > 0.05 → cannot reject normality
        }
    )

# Convert to DataFrame
normality_report = pd.DataFrame(normality_results)

# Display report
print("Normality Check Report (Shapiro-Wilk Test):")
print(normality_report)

# Save the report in a variable for later use
report_variable = normality_report
