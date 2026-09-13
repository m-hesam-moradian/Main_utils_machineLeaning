import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_QR"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column]).values
y = df[target_column].values

# --- Standardize Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split into train/test (80/20, shuffle=False to match K-Fold) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# --- Train Quantile Regression Model ---
model = HistGradientBoostingRegressor(
    loss='quantile',
    quantile=0.5,
    max_iter=90,
    min_samples_leaf=20,
    random_state=45
)

model.fit(X_train, y_train)

# --- Predictions ---
pred_raw_train = model.predict(X_train)
pred_raw_test = model.predict(X_test)
pred_raw_all = model.predict(X_scaled)

y_pred_train = np.clip(np.round(pred_raw_train), 0, 2).astype(int)
y_pred_test = np.clip(np.round(pred_raw_test), 0, 2).astype(int)
y_pred_all = np.clip(np.round(pred_raw_all), 0, 2).astype(int)

# --- Accuracy metrics ---
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
acc_all = accuracy_score(y, y_pred_all)

print("[QR] Quantile Regression Accuracy")
print("---------------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")

# --- Generate Probability distribution based on distance to classes ---
classes = np.unique(y)
distances = np.abs(pred_raw_all[:, None] - classes[None, :])
inv_dist = 1.0 / (distances + 0.1)
y_pred_proba = inv_dist / inv_dist.sum(axis=1, keepdims=True)

proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in classes]
)

# Combine results
df_all = pd.concat([
    pd.DataFrame({"y_real": y, "y_pred": y_pred_all}),
    proba_df
], axis=1)

# Export to .npt files
df_all.to_csv(r"data/model_QR.npt", sep="\t", index=False, header=False)
print("Saved predictions to data/model_QR.npt")
