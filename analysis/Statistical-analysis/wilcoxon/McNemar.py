from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
import numpy as np
from itertools import combinations

# Load data
df = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    header=0,
    sheet_name="predicts",
)

# --- THE FIX IS HERE ---

# 1. True labels: We only need the very first column since all 'Real' columns are the same
y_true = df.iloc[:, 0]

# 2. Predictions: Grab every second column starting from index 1 (skipping the extra 'Real' columns)
preds = df.iloc[:, 1::2]

# 3. Clean names: Grab the clean model names from the even columns and overwrite the ".1" names
preds.columns = df.columns[0::2]

# -----------------------

results = []

# Compare every pair of models
for col1, col2 in combinations(preds.columns, 2):

    m1_correct = preds[col1] == y_true
    m2_correct = preds[col2] == y_true

    # Full 2x2 contingency table
    a = np.sum(m1_correct & m2_correct)       # both correct
    b = np.sum(m1_correct & ~m2_correct)      # model1 correct, model2 wrong
    c = np.sum(~m1_correct & m2_correct)      # model1 wrong, model2 correct
    d = np.sum(~m1_correct & ~m2_correct)     # both wrong

    table = [
        [a, b],
        [c, d]
    ]

    # Handle identical predictions
    if b + c == 0:
        statistic = np.nan
        p_value = 1.0
    else:
        result = mcnemar(
            table,
            exact=False,
            correction=True
        )
        statistic = result.statistic
        p_value = result.pvalue

    results.append({
        "Model_1": col1,
        "Model_2": col2,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "Statistic": statistic,
        "p_value": p_value
    })

# Save results
results_df = pd.DataFrame(results)

results_df.to_clipboard(index=False)

print(results_df)
print("\n✅ Success! Results copied to clipboard with clean model names.")