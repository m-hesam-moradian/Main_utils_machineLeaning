import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from itertools import combinations

# Load structured data from Excel
df = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    # r"C:\Users\Sam\Downloads\Telegram Desktop\749.xlsx",
    header=0,
    sheet_name="predicts(ENN)",
)
  
# Dynamically extract model names and predictions
columns = df.columns.tolist()
structured_data = []

for i in range(0, len(columns), 2):
    name = columns[i].strip()
    y_real = df.iloc[:, i].tolist()
    y_predict = df.iloc[:, i + 1].tolist()
    structured_data.append({"name": name, "y_real": y_real, "y_predict": y_predict})

# Build prediction dictionary for the T-test
predictions = {entry["name"]: np.array(entry["y_predict"]) for entry in structured_data}

# Initialize results dictionary
results = {
    "stats": {},
    "p_values": {},
}

# Set your significance level
alpha = 0.05

# Perform Paired T-Test (ttest_rel) for all unique model pairs
for model_a, model_b in combinations(predictions.keys(), 2):
    try:
        # Changed from wilcoxon() to ttest_rel()
        t_stat, p_value = ttest_rel(predictions[model_a], predictions[model_b])
        results["stats"][f"{model_a} vs {model_b}"] = t_stat
        results["p_values"][f"{model_a} vs {model_b}"] = p_value
    except Exception as e:
        results["stats"][f"{model_a} vs {model_b}"] = np.nan
        results["p_values"][f"{model_a} vs {model_b}"] = np.nan
        print(f"Error comparing {model_a} vs {model_b}: {e}")

# Convert results to DataFrame
# Changed column name from "Statistic" to "t-statistic" to reflect the new math
df_stats = pd.DataFrame(results["stats"].items(), columns=["Comparison", "t-statistic"])
df_p_values = pd.DataFrame(results["p_values"].items(), columns=["Comparison", "P-Value"])

# Merge the tables
df_results = pd.merge(df_stats, df_p_values, on="Comparison")

# --- NEW FEATURE: Add Significance Text ---
# This applies the logic from your sample code directly into a new column
def check_significance(p):
    if pd.isna(p):
        return "NaN"
    return "Significant" if p < alpha else "Not Significant"

df_results["Result (alpha=0.05)"] = df_results["P-Value"].apply(check_significance)
# ------------------------------------------

# Force the P-Values to display as standard decimals with 15 decimal places
df_results["P-Value"] = df_results["P-Value"].apply(lambda x: f"{x:.15f}" if pd.notna(x) else "NaN")

# Display final merged results
print("\nPaired T-Test Comparison Results:")
print(df_results)

# Optional: send to clipboard if you want to paste it straight into Excel
df_results.to_clipboard()