import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# --- Toggle for Ablation Study ---
USE_SMOTE = False
tag = "SMOTE" if USE_SMOTE else "No_SMOTE"

# --- Load Excel file ---
excel_path = r"D:\ML\task\Data.xlsx"
sheet_name = f"Data_after_KFold_ETC({tag})"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Split into train/test (80/20, shuffle=False) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

if USE_SMOTE:
    smote = SMOTE(random_state=42)
    X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
else:
    X_train_fit, y_train_fit = X_train, y_train

# --- Train ETC (hyperparams synced from K-Fold CV) ---
model = ExtraTreesClassifier(
    n_estimators=30,
    max_depth=4,
    min_samples_split=30,
    min_samples_leaf=15,
    max_features=0.4,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_fit, y_train_fit)

# --- Predictions ---
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)
y_pred_all   = model.predict(X)

# --- Accuracy metrics ---
acc_train = accuracy_score(y_train, y_pred_train)
acc_test  = accuracy_score(y_test,  y_pred_test)
acc_all   = accuracy_score(y,       y_pred_all)

print(f"[+] ETC Accuracy Results ({tag})")
print("----------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")
print(f"Precision (test)  : {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"Recall    (test)  : {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"F1-Score  (test)  : {f1_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"MCC       (test)  : {matthews_corrcoef(y_test, y_pred_test):.4f}")

# --- Export to data/model4.npt (Model 2 Slot for single base) ---
# Note: LGBC is model 1, ETC is model 4, BC is model 7 in the final scheme.
df_all = pd.DataFrame({"y_real": y.values, "y_pred": y_pred_all})

out_dir = rf"D:\ML\data\{tag}"
os.makedirs(out_dir, exist_ok=True)
np.savetxt(os.path.join(out_dir, "model4.npt"), df_all.values, fmt="%d")
print(f"\n[+] Saved: {out_dir}/model4.npt")