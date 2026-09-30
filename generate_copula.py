import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_file = r"d:\ML\task\Data.xlsx"
print("Reading Tabular_Copula sheet...")
df = pd.read_excel(data_file, sheet_name="Tabular_Copula", index_col=0)

plt.figure(figsize=(16, 14))
sns.heatmap(df, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, square=True, annot_kws={"size": 8})
plt.title("Tabular Copula Sensitivity Plot (Regression - MSE)", fontsize=18)
plt.xlabel("Feature 2 (Perturbed by 10%)", fontsize=14)
plt.ylabel("Feature 1 (Perturbed by 10%)", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

output_path = r"d:\ML\copula_heatmap_readable.png"
plt.savefig(output_path, dpi=300)
print(f"Saved correctly formatted copula heatmap to {output_path}")
