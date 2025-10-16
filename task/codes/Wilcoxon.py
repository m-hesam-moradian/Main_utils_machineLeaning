import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

# Load structured data from Excel
df = pd.read_excel(
    r"D:\ML\Main_utils_machineLeaning\task\BSE. No.13-Dataset.xlsx",
    header=0,
    sheet_name="Sheet2",
)

# Dynamically extract model names and predictions
columns = df.columns.tolist()
structured_data = []

for i in range(0, len(columns), 2):
    name = columns[i].strip()
    y_real = df.iloc[:, i].tolist()
    y_predict = df.iloc[:, i + 1].tolist()
    structured_data.append({"name": name, "y_real": y_real, "y_predict": y_predict})

# Build prediction dictionary for Wilcoxon
predictions = {entry["name"]: np.array(entry["y_predict"]) for entry in structured_data}

# Initialize results dictionary
results = {
    "stats": {},
    "p_values": {},
}

# Perform Wilcoxon signed-rank test for all unique model pairs
for model_a, model_b in combinations(predictions.keys(), 2):
    try:
        stat, p_value = wilcoxon(predictions[model_a], predictions[model_b])
        results["stats"][f"{model_a} vs {model_b}"] = stat
        results["p_values"][f"{model_a} vs {model_b}"] = p_value
    except Exception as e:
        results["stats"][f"{model_a} vs {model_b}"] = np.nan
        results["p_values"][f"{model_a} vs {model_b}"] = np.nan
        print(f"Error comparing {model_a} vs {model_b}: {e}")

# Print summary
for key, value in results.items():
    print(f"\n{key}:")
    for sub_key, sub_value in value.items():
        print(
            f"  {sub_key}: {sub_value:.5f}"
            if not pd.isna(sub_value)
            else f"  {sub_key}: NaN"
        )

# Convert results to DataFrame
df_stats = pd.DataFrame(results["stats"].items(), columns=["Comparison", "Statistic"])
df_p_values = pd.DataFrame(
    results["p_values"].items(), columns=["Comparison", "P-Value"]
)
df_results = pd.merge(df_stats, df_p_values, on="Comparison")

# Display final merged results
print("\nWilcoxon Comparison Results:")
print(df_results)

# Optional: significance check for one pair
alpha = 0.05
first_pair = list(results["p_values"].keys())[0]
if results["p_values"][first_pair] < alpha:
    print(f"\n{first_pair} shows a significant difference.")
else:
    print(f"\nNo significant difference between {first_pair}.")
