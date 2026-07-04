import pandas as pd
import numpy as np
from scipy.stats import shapiro


def shapiro_wilk_test(df, alpha=0.05):
    """
    Performs the Shapiro-Wilk normality test on all numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    alpha : float
        Significance level.

    Returns
    -------
    report_df : pandas.DataFrame
        Summary report of the Shapiro-Wilk test.
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    report_data = []

    for col in numeric_cols:

        # Remove missing values
        data = df[col].dropna()

        # Shapiro requires at least 3 samples
        if len(data) < 3:
            report_data.append({
                "Feature": col,
                "Samples": len(data),
                "Statistic": np.nan,
                "P-Value": np.nan,
                "Normal": "Insufficient Data",
                "Conclusion": "Less than 3 samples"
            })
            continue

        statistic, p_value = shapiro(data)

        normal = "Yes" if p_value > alpha else "No"

        conclusion = (
            "Normally Distributed"
            if p_value > alpha
            else "Not Normally Distributed"
        )

        print(
            f"{col}: Statistic={statistic:.4f}, "
            f"P-value={p_value:.4f} --> {conclusion}"
        )

        report_data.append({
            "Feature": col,
            "Samples": len(data),
            "Statistic": round(statistic, 6),
            "P-Value": round(p_value, 6),
            "Normal": normal,
            "Conclusion": conclusion
        })

    report_df = pd.DataFrame(report_data)

    return report_df


# ==========================
# CONFIGURATION
# ==========================

input_file = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
input_sheet = "Z-Score"

report_sheet = "Shapiro_Report"

# ==========================
# LOAD DATA
# ==========================

df = pd.read_excel(input_file, sheet_name=input_sheet)

# ==========================
# RUN SHAPIRO TEST
# ==========================

report_df = shapiro_wilk_test(df)

# ==========================
# SAVE REPORT
# ==========================

with pd.ExcelWriter(
    input_file,
    mode="a",
    engine="openpyxl",
    if_sheet_exists="replace"
) as writer:

    report_df.to_excel(writer, sheet_name=report_sheet, index=False)

print(f"✅ Shapiro-Wilk report saved to '{report_sheet}'")