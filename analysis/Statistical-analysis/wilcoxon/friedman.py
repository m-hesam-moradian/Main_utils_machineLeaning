import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from itertools import combinations

# Load structured data from Excel
df = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    header=0,
    sheet_name="Wilcoxon test Data",
)
  
# Dynamically extract model names and predictions
columns = df.columns.tolist()
structured_data =[]

for i in range(0, len(columns), 2):
    name = columns[i].strip()
    y_real = df.iloc[:, i].tolist()
    # Drop NaNs so the Friedman test doesn't crash
    y_predict = df.iloc[:, i + 1].dropna().tolist()
    structured_data.append({"name": name, "y_real": y_real, "y_predict": y_predict})

# Build prediction dictionary 
predictions = {entry["name"]: np.array(entry["y_predict"]) for entry in structured_data}

# Ensure all prediction arrays have the same length (Friedman is strict about this)
min_length = min([len(v) for v in predictions.values()])
for name in predictions:
    predictions[name] = predictions[name][:min_length]

# Initialize results dictionary
results = {
    "stats": {},
    "p_values": {},
}

# Perform Friedman test for combinations of 3 models to fix the error!
for model_a, model_b, model_c in combinations(predictions.keys(), 3):
    comparison_name = f"{model_a} vs {model_b} vs {model_c}"
    try:
        # Pass 3 models to friedmanchisquare
        stat, p_value = friedmanchisquare(
            predictions[model_a], 
            predictions[model_b], 
            predictions[model_c]
        )
        results["stats"][comparison_name] = stat
        results["p_values"][comparison_name] = p_value
    except Exception as e:
        results["stats"][comparison_name] = np.nan
        results["p_values"][comparison_name] = np.nan
        print(f"Error comparing {comparison_name}: {e}")

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
print("\nFriedman 3-Way Comparison Results:")
print(df_results)

# Optional: significance check for one group
alpha = 0.05
first_group = list(results["p_values"].keys())[0]
if results["p_values"][first_group] < alpha:
    print(f"\n{first_group} shows a significant difference.")
else:
    print(f"\nNo significant difference between {first_group}.")

# Copies the 3-way comparisons to your clipboard automatically!
df_results.to_clipboard(index=False)
print("\n[ Success: The Friedman results have been copied to your clipboard! ]")