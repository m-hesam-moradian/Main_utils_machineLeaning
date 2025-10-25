import pandas as pd
import numpy as np
import smogn

# -------------------- 1. Load your dataset --------------------
file_path = r"C:\Users\Sam\Desktop\ML\task\BSS.No.1-Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="Z-Score")

# -------------------- 2. Define CPU usage column --------------------
cpu_col = "CPU Usage"  # Replace with your actual column name

# -------------------- 3. Bin CPU usage into Low, Medium, High --------------------
bins = [0, 20, 60, 100]
labels = ["Low", "Medium", "High"]
df["CPU_Bin"] = pd.cut(df[cpu_col], bins=bins, labels=labels, include_lowest=True)

# -------------------- 4. Analyze bin distribution --------------------
bin_counts = df["CPU_Bin"].value_counts().sort_index()
bin_percentages = bin_counts / len(df) * 100

print("CPU Usage Distribution:")

print(pd.DataFrame({"Count": bin_counts, "Percentage": bin_percentages.round(2)}))

# -------------------- 5. Check for imbalance --------------------
threshold = 10  # percent
rare_bins = bin_percentages[bin_percentages < threshold]

# -------------------- 6. Apply SMOGN if imbalance is detected --------------------
if not rare_bins.empty:
    print("\n⚠️ Imbalance detected in bins:", list(rare_bins.index))
    print("→ Applying SMOGN to balance rare CPU usage ranges...")

    df_smogn = smogn.smoter(
        data=df.drop(columns=["CPU_Bin"]),
        y=cpu_col,
        k=5,
        samp_method="extreme",
        rel_thres=0.8,
        rel_method="auto",
        under_samp=True
    )
else:
    print("\n✅ No imbalance detected. Skipping SMOGN.")
    df_smogn = df.drop(columns=["CPU_Bin"])

# -------------------- 7. Save to clipboad --------------------
df_smogn.to_clipboard(index=False)