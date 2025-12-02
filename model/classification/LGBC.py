import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
import numpy as np

# --- Load reordered data for LGBC (after K-Fold) ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_LGBC"  # Update if needed

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Use last 20% as test set to match K-Fold logic ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Train and predict ---
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42
)
model.fit(X_train, y_train)

y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Predicted probabilities ---
y_pred_proba_all = model.predict_proba(X)
y_pred_proba_train = model.predict_proba(X_train)
y_pred_proba_test = model.predict_proba(X_test)

# --- Metrics function ---
def get_classification_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    class_error = 1 - np.trace(cm) / np.sum(cm)  # overall misclassification rate
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr") if y_prob is not None else None

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1-Score": f1_score(y_true, y_pred, average="macro"),
        "Class-Wise Error": class_error,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": auc
    }

# --- Evaluate sets ---
mid = len(y_test) // 2
sets = [
    ("All", y, y_pred_all, y_pred_proba_all),
    ("Train", y_train, y_pred_train, y_pred_proba_train),
    ("Test", y_test, y_pred_test, y_pred_proba_test),
    ("Value", y_test[:mid], y_pred_test[:mid], y_pred_proba_test[:mid]),
    ("Test-Value", y_test[mid:], y_pred_test[mid:], y_pred_proba_test[mid:])
]

df_metrics = pd.DataFrame([
    {"Set": s, **get_classification_metrics(y_t, y_p, y_prob)}
    for s, y_t, y_p, y_prob in sets
])

print(df_metrics)

# --- Build probability DataFrames ---
proba_cols = [f"Prob_Class_{cls}" for cls in model.classes_]

df_all = pd.concat([
    pd.DataFrame({"y_real": y, "y_pred": y_pred_all}),
    pd.DataFrame(y_pred_proba_all, columns=proba_cols)
], axis=1)

df_train = pd.concat([
    pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train}),
    pd.DataFrame(y_pred_proba_train, columns=proba_cols)
], axis=1)

df_test = pd.concat([
    pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test}),
    pd.DataFrame(y_pred_proba_test, columns=proba_cols)
], axis=1)

# --- Optional: view first few rows ---
print("\n📊 Sample of overall predictions with probabilities:")
print(df_all.head())

# --- Export to clipboard ---
df_all.to_clipboard(index=False, header=False)