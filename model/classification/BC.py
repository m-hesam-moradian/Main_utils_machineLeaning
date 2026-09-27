import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
import os

# --- Load reordered data (after K-Fold) ---
excel_path = r"D:\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_BC(SMOTE)"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]

# Prepare the features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Use last 20% as test set ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Initialize model ---
model = ExtraTreesClassifier(
    n_estimators=5,
    max_depth=4,
    min_samples_split=30,
    min_samples_leaf=15,
    max_features=0.4,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Metrics ---
mid = len(y_test) // 2
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test[:mid], y_pred_test[:mid]),
    ("Test-Value", y_test[mid:], y_pred_test[mid:])
]

df_metrics = pd.DataFrame([{
    "Set": s,
    "Accuracy": accuracy_score(y_t, y_p),
    "F1 Score": f1_score(y_t, y_p, average='weighted'),
    "Precision": precision_score(y_t, y_p, average='weighted', zero_division=0)
} for s, y_t, y_p in sets])

print(df_metrics)

# --- Output predictions ---
y_prob_all = model.predict_proba(X)
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
for i in range(y_prob_all.shape[1]):
    df_all[f"prob_{i}"] = y_prob_all[:, i]

df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# --- Export to clipboard & file ---
df_all.to_clipboard(index=False, header=False)

out_dir = r"data\SMOTE"
os.makedirs(out_dir, exist_ok=True)
np.savetxt(os.path.join(out_dir, "model7.npt"), df_all.values, fmt="%f")
print(f"\n[+] Saved: {out_dir}\\model7.npt")
