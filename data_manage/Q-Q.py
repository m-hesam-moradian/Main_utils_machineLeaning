import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

def generate_qq_plots(df, save_folder="QQ_Plots"):
    """
    Generates a Q-Q plot for every numeric column.
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    os.makedirs(save_folder, exist_ok=True)

    report_data = []

    for col in numeric_cols:

        data = df[col].dropna()

        plt.figure(figsize=(6,6))
        sm.qqplot(data, line='45', fit=True)

        plt.title(f"Q-Q Plot - {col}")
        plt.grid(alpha=0.3)

        filename = os.path.join(save_folder, f"{col}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        report_data.append({
            "Feature": col,
            "Samples": len(data),
            "Plot Saved": filename
        })

        print(f"✓ {col}")

    report_df = pd.DataFrame(report_data)

    return report_df


# ================= CONFIGURATION =================

input_file = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

input_sheet = "Z-Score"

report_sheet = "QQ_Report"

plot_folder = r"C:\Users\Sam\Desktop\ML\task\QQ_Plots"

# ================= PROCESS =================

df = pd.read_excel(input_file, sheet_name=input_sheet)

report_df = generate_qq_plots(df, plot_folder)

# ================= SAVE REPORT =================

with pd.ExcelWriter(
        input_file,
        mode="a",
        engine="openpyxl",
        if_sheet_exists="replace"
) as writer:

    report_df.to_excel(writer,
                       sheet_name=report_sheet,
                       index=False)

print("\n✅ All Q-Q plots generated.")
print(f"✅ Report saved to sheet '{report_sheet}'")
print(f"✅ Plots saved in:\n{plot_folder}")