import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from itertools import combinations
import mpmath as mp


# ============================================================
# High-precision settings
# ============================================================

mp.mp.dps = 100  # Number of decimal digits of precision


# ============================================================
# Function to format very small p-values
# ============================================================

def format_p_value(p_value):
    """
    Format p-values in scientific notation.

    Very small p-values that would normally become 0.0
    are represented using high-precision scientific notation.
    """

    if p_value is None or pd.isna(p_value):
        return "NaN"

    # Convert scipy float to high precision
    p_mp = mp.mpf(str(p_value))

    if p_mp == 0:
        return "0"

    # Scientific notation
    exponent = int(mp.floor(mp.log10(p_mp)))
    mantissa = p_mp / mp.power(10, exponent)

    return f"{float(mantissa):.3f}E{exponent:+d}"


# ============================================================
# High-precision Friedman p-value for 3 models
# ============================================================

def high_precision_friedman_pvalue(statistic, num_models=3):
    """
    Calculate Friedman test p-value with high precision.

    For 3 models:
        df = k - 1 = 2

    For chi-square distribution with df=2:
        p = exp(-statistic / 2)

    This avoids scipy's floating-point underflow to 0.
    """

    df = num_models - 1

    statistic_mp = mp.mpf(str(statistic))

    # For df = 2, survival function has a simple closed form
    if df == 2:
        p_value = mp.exp(-statistic_mp / 2)

    else:
        # General case using regularized upper incomplete gamma
        p_value = mp.gammainc(
            mp.mpf(df) / 2,
            statistic_mp / 2,
            mp.inf,
            regularized=True
        )

    return p_value


# ============================================================
# Load structured data from Excel
# ============================================================

df = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    header=0,
    sheet_name="predicts",
)


# ============================================================
# Dynamically extract model names and predictions
# ============================================================

columns = df.columns.tolist()

structured_data = []

for i in range(0, len(columns), 2):

    name = str(columns[i]).strip()

    y_real = df.iloc[:, i].dropna().tolist()

    y_predict = df.iloc[:, i + 1].dropna().tolist()

    structured_data.append({
        "name": name,
        "y_real": y_real,
        "y_predict": y_predict
    })


# ============================================================
# Build prediction dictionary
# ============================================================

predictions = {
    entry["name"]: np.array(entry["y_predict"], dtype=float)
    for entry in structured_data
}


# ============================================================
# Ensure all prediction arrays have the same length
# ============================================================

min_length = min(
    len(values)
    for values in predictions.values()
)

for name in predictions:
    predictions[name] = predictions[name][:min_length]


# ============================================================
# Initialize results
# ============================================================

results = {
    "stats": {},
    "p_values": {},
}


# ============================================================
# Friedman 3-way comparisons
# ============================================================

print("\nRunning Friedman 3-Way Comparisons...")
print("=" * 70)

for model_a, model_b, model_c in combinations(
    predictions.keys(), 3
):

    comparison_name = (
        f"{model_a} vs {model_b} vs {model_c}"
    )

    try:

        # ----------------------------------------------------
        # Friedman test
        # ----------------------------------------------------

        statistic, scipy_p_value = friedmanchisquare(
            predictions[model_a],
            predictions[model_b],
            predictions[model_c]
        )

        # ----------------------------------------------------
        # High-precision p-value
        # ----------------------------------------------------

        high_precision_p = high_precision_friedman_pvalue(
            statistic,
            num_models=3
        )

        results["stats"][comparison_name] = statistic

        # Store high precision value as mpmath object
        results["p_values"][comparison_name] = high_precision_p

    except Exception as e:

        results["stats"][comparison_name] = np.nan
        results["p_values"][comparison_name] = None

        print(
            f"Error comparing {comparison_name}: {e}"
        )


# ============================================================
# Create results DataFrame
# ============================================================

result_rows = []

for comparison in results["stats"]:

    statistic = results["stats"][comparison]
    p_value = results["p_values"][comparison]

    if p_value is not None:
        formatted_p = format_p_value(p_value)
    else:
        formatted_p = "NaN"

    result_rows.append({
        "Comparison": comparison,
        "Statistic": statistic,
        "P-Value": formatted_p
    })


df_results = pd.DataFrame(result_rows)


# ============================================================
# Format statistic
# ============================================================

df_results["Statistic"] = df_results["Statistic"].apply(
    lambda x: f"{x:.5f}" if pd.notna(x) else "NaN"
)


# ============================================================
# Display results
# ============================================================

print("\n")
print("=" * 100)
print("FRIEDMAN 3-WAY COMPARISON RESULTS")
print("=" * 100)

print(
    df_results.to_string(index=False)
)


# ============================================================
# Significance check
# ============================================================

alpha = 0.05

print("\n")
print("=" * 70)
print("SIGNIFICANCE RESULTS")
print("=" * 70)

for comparison, p_value in results["p_values"].items():

    if p_value is not None:

        if p_value < alpha:
            print(
                f"✓ Significant: {comparison}"
            )
        else:
            print(
                f"✗ Not significant: {comparison}"
            )




# ============================================================
# Copy results to clipboard
# ============================================================

df_results.to_clipboard(index=False)

print("\n")
print("=" * 70)
print("✓ Results copied to clipboard")
print("=" * 70)