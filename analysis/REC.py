import numpy as np
import pandas as pd
from sklearn.metrics import auc

# -------------------- 1. Load prediction data --------------------
data = np.loadtxt(r"C:\Users\Sam\Desktop\ML\data\Data_err.npt")
y_real = data[:, 0]
y_pred = data[:, 1]

# -------------------- 2. Compute absolute errors --------------------
errors = np.abs(y_real - y_pred)
epsilon = np.linspace(0, errors.max(), 200)
accuracy = [np.mean(errors <= e) for e in epsilon]
# -------------------- 3. Compute AUC --------------------
rec_auc = auc(epsilon, accuracy)

# -------------------- 4. Create REC DataFrame --------------------
df_rec = pd.DataFrame({
    "Epsilon": epsilon,
    "Accuracy": accuracy,
    "AUC": ["" for _ in range(len(epsilon))]  # Fill AUC column with empty strings
})

# Append AUC value in the last row
df_rec.loc[0] = ["", "", rec_auc]

# -------------------- 5. Output --------------------
print(df_rec.tail(10))  # Preview last rows including AUC
df_rec.to_clipboard(index=False)
# Optional: df_rec.to_excel("REC_Curve_with_AUC.xlsx", index=False)