import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load your data
data = np.loadtxt(r"D:\ML\ML\data\Data_err.npt")
y = data[:, 0]
predictData = data[:, 1]

# Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y, predictData)
roc_auc = auc(fpr, tpr)

# Create AUC column: empty strings except last row
auc_column = [""] * (len(fpr) - 1) + [round(roc_auc, 3)]

# Build DataFrame
roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "AUC": auc_column})

# Show the table
print(roc_df)


# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
