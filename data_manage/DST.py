import numpy as np
import pandas as pd

# -------------------- 1. Load data --------------------
data = np.loadtxt(r"C:\Users\Sam\Desktop\ML\data\Data_err.npt")
y_real = data[:, 0]
y_pred_lgbr = data[:, 1]
y_pred_sgb = data[:, 2]

# -------------------- 2. Compute model errors --------------------
error_lgbr = np.abs(y_real - y_pred_lgbr)
error_sgb = np.abs(y_real - y_pred_sgb)

# Avoid division by zero
error_lgbr += 1e-8
error_sgb += 1e-8

# -------------------- 3. Compute belief weights --------------------
belief_lgbr = 1 / error_lgbr
belief_sgb = 1 / error_sgb
total_belief = belief_lgbr + belief_sgb

# Normalize to get mass functions
m_lgbr = belief_lgbr / total_belief
m_sgb = belief_sgb / total_belief

# -------------------- 4. DST Fusion --------------------
y_pred_dst = m_lgbr * y_pred_lgbr + m_sgb * y_pred_sgb

# -------------------- 5. Output fused predictions --------------------
df_fused = pd.DataFrame({
    "y_real": y_real,
    "y_pred_KNNC": y_pred_lgbr,
    "y_pred_XGB": y_pred_sgb,
    "y_pred_DST": y_pred_dst
})

print(df_fused.head())
df_fused.to_clipboard(index=False)
# Optional: df_fused.to_excel("DST_Fused_Predictions.xlsx", index=False)