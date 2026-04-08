import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

# --- Load fold-wise metrics from Excel ---
df = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    sheet_name="predicts",  # Each column = model, each row = fold's F1-score
)

# --- Extract model names and their F1-score arrays ---
model_names = df.columns.tolist()
metrics = {model: df[model].dropna().values for model in model_names}

# --- Initialize results ---
results = {
    "stats": {},
    "p_values": {},
}

# --- Pairwise Wilcoxon signed-rank test ---
for model_a, model_b in combinations(model_names, 2):
    try:
        stat, p_value = wilcoxon(metrics[model_a], metrics[model_b])
        results["stats"][f"{model_a} vs {model_b}"] = stat
        results["p_values"][f"{model_a} vs {model_b}"] = p_value
    except Exception as e:
        results["stats"][f"{model_a} vs {model_b}"] = np.nan
        results["p_values"][f"{model_a} vs {model_b}"] = np.nan
        print(f"Error comparing {model_a} vs {model_b}: {e}")

# --- Print summary ---
for key, value in results.items():
    print(f"\n{key}:")
    for sub_key, sub_value in value.items():
        print(
            f"  {sub_key}: {sub_value:.5f}"
            if not pd.isna(sub_value)
            else f"  {sub_key}: NaN"
        )

# --- Merge into DataFrame ---
df_stats = pd.DataFrame(results["stats"].items(), columns=["Comparison", "Statistic"])
df_p_values = pd.DataFrame(results["p_values"].items(), columns=["Comparison", "P-Value"])
df_results = pd.merge(df_stats, df_p_values, on="Comparison")

# --- Display final results ---
print("\n📊 Wilcoxon Test: Comparing Classification Models by F1-Score Across Folds")
print(df_results)

# --- Optional: significance check for first pair ---
alpha = 0.05
first_pair = list(results["p_values"].keys())[0]
if results["p_values"][first_pair] < alpha:
    print(f"\n✅ {first_pair} shows a statistically significant difference (p < {alpha}).")
else:
    print(f"\n❌ No significant difference between {first_pair} (p ≥ {alpha}).")

# --- Export to clipboard ---
df_results.to_clipboard(index=False)