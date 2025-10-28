import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# -------------------- 1. Load and prepare data --------------------
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.21-Data.xlsx", sheet_name="Data_after_KFold_LGBR")
y_real = df["SOH"].astype(float).values
X_raw = pd.get_dummies(df.drop(columns=["SOH"]), drop_first=True)
X = StandardScaler().fit_transform(X_raw)

# -------------------- 2. Train base models --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_real, test_size=0.3, shuffle=False, random_state=42)

model_lgbr = LGBMRegressor()
model_sgb = GradientBoostingRegressor()

model_lgbr.fit(X_train, y_train)
model_sgb.fit(X_train, y_train)

# -------------------- 3. DST Fusion --------------------
y_pred_lgbr = model_lgbr.predict(X)
y_pred_sgb = model_sgb.predict(X)

error_lgbr = np.abs(y_real - y_pred_lgbr) + 1e-8
error_sgb = np.abs(y_real - y_pred_sgb) + 1e-8

belief_lgbr = 1 / error_lgbr
belief_sgb = 1 / error_sgb
total_belief = belief_lgbr + belief_sgb

m_lgbr = belief_lgbr / total_belief
m_sgb = belief_sgb / total_belief

y_pred_dst = m_lgbr * y_pred_lgbr + m_sgb * y_pred_sgb

# -------------------- 4. Monte Carlo Simulation --------------------
n_simulations = 1000
noise_std = 0.01  # Adjust based on feature sensitivity

def simulate_model(model, X, n_sim=1000, noise_std=0.01):
    preds = []
    for _ in range(n_sim):
        X_noisy = X + np.random.normal(0, noise_std, X.shape)
        preds.append(model.predict(X_noisy))
    return np.array(preds)

# Simulate LGBR and SGB
sim_lgbr = simulate_model(model_lgbr, X, n_simulations, noise_std)
sim_sgb = simulate_model(model_sgb, X, n_simulations, noise_std)

# Simulate DST by fusing each simulation
sim_dst = (sim_lgbr + sim_sgb) / 2  # or apply DST weights per simulation if needed

# -------------------- 5. Compute uncertainty metrics --------------------
def summarize_mcs(simulated_preds, y_true):
    mean_pred = simulated_preds.mean(axis=0)
    std_pred = simulated_preds.std(axis=0)
    lower = np.percentile(simulated_preds, 2.5, axis=0)
    upper = np.percentile(simulated_preds, 97.5, axis=0)
    return pd.DataFrame({
        "y_real": y_true,
        "mean_pred": mean_pred,
        "std_pred": std_pred,
        "CI_lower": lower,
        "CI_upper": upper
    })

df_lgbr = summarize_mcs(sim_lgbr, y_real)
df_sgb = summarize_mcs(sim_sgb, y_real)
df_dst = summarize_mcs(sim_dst, y_real)

# -------------------- 6. Output --------------------
print("LGBR Uncertainty Sample:")
print(df_lgbr.head())

print("\nSGB Uncertainty Sample:")
print(df_sgb.head())

print("\nDST Uncertainty Sample:")
print(df_dst.head())

# Optional: Export to Excel or clipboard
# df_lgbr.to_excel("LGBR_Uncertainty.xlsx", index=False)
# df_sgb.to_excel("SGB_Uncertainty.xlsx", index=False)
# df_dst.to_excel("DST_Uncertainty.xlsx", index=False)