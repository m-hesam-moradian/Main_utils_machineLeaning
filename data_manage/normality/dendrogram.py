import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# Load the dataset
file_path = (
    r"D:\ML\Main_utils\Task\Original Dataset- Concrete (elevated temperature).xlsx"
)
sheet_name = "Standard_Normalized"
dt = pd.read_excel(file_path, sheet_name=sheet_name)

# Standardize the features (optional but recommended for clustering)
df_standardized = (dt - dt.mean()) / dt.std()

# Compute the hierarchical clustering
linked = linkage(df_standardized.T, method="ward")  # Transpose to cluster features

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked, labels=dt.columns, leaf_rotation=90, leaf_font_size=10)
plt.title("Dendrogram for Feature Clustering EGB")
plt.xlabel("Features")
plt.ylabel("Distance")

# Save as PNG in the same folder as the Excel file
output_dir = os.path.dirname(file_path)
output_file = os.path.join(output_dir, "dendrogram_features.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"✅ Dendrogram saved as PNG: {output_file}")
