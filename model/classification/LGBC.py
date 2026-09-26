import numpy as np
from lightgbm import LGBMClassifier
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
sheet_name = f"Data_after_KFold_LGBC({tag})"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Split into train/test (80/20, shuffle=False) ---
# Best fold test set is already placed in last 20% by K-Fold script
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

if USE_SMOTE:
    smote = SMOTE(random_state=42)
    X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
else:
    X_train_fit, y_train_fit = X_train, y_train

# --- Train LGBC (hyperparams synced from K-Fold CV) ---
model = LGBMClassifier(
    n_estimators=50,
    learning_rate=0.005,
    max_depth=5,
    num_leaves=31,
    min_child_samples=80,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
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

print(f"[+] LGBC Accuracy Results ({tag})")
print("----------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")
print(f"Precision (test)  : {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"Recall    (test)  : {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"F1-Score  (test)  : {f1_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"MCC       (test)  : {matthews_corrcoef(y_test, y_pred_test):.4f}")

# --- Build df_all (y_real, y_pred) for npt export ---
df_all   = pd.DataFrame({"y_real": y.values,       "y_pred": y_pred_all})
df_train = pd.DataFrame({"y_real": y_train.values, "y_pred": y_pred_train})
df_test  = pd.DataFrame({"y_real": y_test.values,  "y_pred": y_pred_test})

# --- Export to data/model1.npt ---
out_dir = rf"D:\ML\data\{tag}"
os.makedirs(out_dir, exist_ok=True)
np.savetxt(os.path.join(out_dir, "model1.npt"), df_all.values, fmt="%d")
np.savetxt(os.path.join(out_dir, "Data_err.npt"), df_all.values, fmt="%d")
print(f"\n[+] Saved: {out_dir}\\model1.npt  |  {out_dir}\\Data_err.npt")

# --- Preview ---
print("\n[+] Sample predictions (head):")
print(df_all.head(10))

# --- Copy to clipboard ---
df_all.to_clipboard(index=False, header=False)
print("[+] Copied to clipboard.")