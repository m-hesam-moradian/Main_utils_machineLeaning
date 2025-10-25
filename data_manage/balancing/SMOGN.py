import pandas as pd
import smogn

# Load your data
file_path = r"C:\Users\Sam\Desktop\ML\task\BSS.No.1-Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="BSS.No.1-Target 1")

# Apply SMOGN
df_smogn = smogn.smoter(
    data=df,
    y="CPU_Usage",          # Replace with your actual target column
    k=5,                    # Number of neighbors
    samp_method="extreme",  # Focus on rare extremes
    rel_thres=0.8,          # Relevance threshold (from paper)
    rel_method="auto",      # Automatically estimate relevance
    under_samp=True         # Enable under-sampling of normal cases
)

# Save to Excel
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl") as writer:
    df_smogn.to_excel(writer, sheet_name="Balanced_SMOGN", index=False)