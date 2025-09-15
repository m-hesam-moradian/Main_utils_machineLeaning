import pandas as pd
from scipy.stats import wilcoxon
import numpy as np

# Load the dataset
df = pd.read_excel(
    r"D:\ML\#M(XGBC&RFC)#O(LEOA)#RTIME#CI#WILCOXONN#FAST\#M(XGBC&RFC)#O(LEOA)#RTIME#CI#WILCOXONN#FAST\data\data.xlsx",
    sheet_name="predicts",
)

# Define the groups of columns based on their base methods
group_cols = {
    "main": [
        "GBC",
        "GBC Optimizer (PVSA)",
        "LGBC",
        "LGBC Optimizer (PVSA)",
        "SC",
        "SC Optimizer (PVSA)",
    ],
}

# List to store results
results = []

# Loop through each group and perform Wilcoxon test on all pairs within the group
for group, cols in group_cols.items():
    # Ensure the columns exist in the dataset
    existing_cols = [col for col in cols if col in df.columns]
    # Generate all unique pairs within this group
    from itertools import combinations

    pairs = list(combinations(existing_cols, 2))

    for col1, col2 in pairs:
        # Extract the samples from the dataset columns
        sample1 = df[col1].dropna()  # Remove any NaN values
        sample2 = df[col2].dropna()  # Remove any NaN values

        # Ensure samples are the same length by aligning them (if needed)
        min_length = min(len(sample1), len(sample2))
        sample1 = sample1[:min_length]
        sample2 = sample2[:min_length]

        # Calculate differences (optional, included as per original code)
        differences = sample1 - sample2

        # Perform the Wilcoxon signed-rank test
        stat, p = wilcoxon(sample1, sample2)

        # Store the results
        results.append(
            {
                "Column1": col1,
                "Column2": col2,
                "Statistic": stat,
                "P-value": p,
                #'Differences': differences.tolist()  # Optional: include differences
            }
        )

# Convert results to a DataFrame for better display
results_df = pd.DataFrame(results)

# Output the results
print("Wilcoxon Signed-Rank Test Results:")
for index, row in results_df.iterrows():
    print(f"\nComparison between {row['Column1']} and {row['Column2']}:")
    print(f"  Wilcoxon test statistic: {row['Statistic']}")
    print(f"  p-value: {row['P-value']}")
    # Uncomment the next line if you want to see the differences
    # print(f"  Differences: {row['Differences']}")

# Optionally, save the results to a file
# results_df.to_csv('wilcoxon_results.csv', index=False)
