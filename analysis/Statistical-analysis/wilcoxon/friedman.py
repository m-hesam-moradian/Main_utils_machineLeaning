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

excel_path = r"D:\ML\task\Data.xlsx"
xl = pd.ExcelFile(excel_path)
tags = ["SMOTE-ENC"]

with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    for tag in tags:
        sheet_name = f"predicts({tag})"
        out_sheet = f"Friedman_Stats({tag})"
        
        if sheet_name not in xl.sheet_names:
            print(f"Skipping {tag}, sheet {sheet_name} not found.")
            continue
            
        print(f"\nRunning Friedman 3-Way Comparisons for '{sheet_name}'...")
        df = pd.read_excel(xl, sheet_name=sheet_name)
        
        columns = df.columns.tolist()
        structured_data = []

        for i in range(0, len(columns), 2):
            if i + 1 < len(columns):
                name = str(columns[i]).strip()
                y_real = df.iloc[:, i].dropna().tolist()
                y_predict = df.iloc[:, i + 1].dropna().tolist()
                structured_data.append({"name": name, "y_real": y_real, "y_predict": y_predict})

        predictions = {entry["name"]: np.array(entry["y_predict"], dtype=float) for entry in structured_data}
        
        min_length = min((len(values) for values in predictions.values()), default=0)
        for name in predictions:
            predictions[name] = predictions[name][:min_length]

        results = {"stats": {}, "p_values": {}}

        for model_a, model_b, model_c in combinations(predictions.keys(), 3):
            comparison_name = f"{model_a} vs {model_b} vs {model_c}"
            try:
                statistic, scipy_p_value = friedmanchisquare(
                    predictions[model_a],
                    predictions[model_b],
                    predictions[model_c]
                )
                high_precision_p = high_precision_friedman_pvalue(statistic, num_models=3)
                results["stats"][comparison_name] = statistic
                results["p_values"][comparison_name] = high_precision_p
            except Exception as e:
                results["stats"][comparison_name] = np.nan
                results["p_values"][comparison_name] = None
                print(f"Error comparing {comparison_name}: {e}")

        result_rows = []
        for comparison in results["stats"]:
            statistic = results["stats"][comparison]
            p_value = results["p_values"][comparison]
            formatted_p = format_p_value(p_value) if p_value is not None else "NaN"
            result_rows.append({
                "Comparison": comparison,
                "Statistic": f"{statistic:.5f}" if pd.notna(statistic) else "NaN",
                "P-Value": formatted_p
            })

        df_results = pd.DataFrame(result_rows)
        df_results.to_excel(writer, sheet_name=out_sheet, index=False)
        print(f"[+] Saved Friedman results to: {out_sheet}")