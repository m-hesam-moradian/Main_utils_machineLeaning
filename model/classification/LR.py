import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix

# --- Load reordered data for LR (after K-Fold) ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_LR"


df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# --- Use last 20% as test set to match K-Fold logic ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Train and predict ---
model = LogisticRegression(
max_iter=1000, random_state=42
)
model.fit(X_train, y_train)

y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:, 1]  # For AUC

# --- Metrics ---
def get_classification_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    class_error = (fp + fn) / len(y_true)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Class-Wise Error": class_error,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": auc
    }

mid = len(y_test) // 2
sets = [
    ("All", y, y_pred_all, None),
    ("Train", y_train, y_pred_train, None),
    ("Test", y_test, y_pred_test, y_prob_test),
    ("Value", y_test[:mid], y_pred_test[:mid], y_prob_test[:mid]),
    ("Test-Value", y_test[mid:], y_pred_test[mid:], y_prob_test[mid:])
]

df_metrics = pd.DataFrame([
    {"Set": s, **get_classification_metrics(y_t, y_p, y_prob)}
    for s, y_t, y_p, y_prob in sets
])

print(df_metrics)

# --- Output predictions ---
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# --- Export to clipboard ---
df_all.to_clipboard(index=False,header=False)
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)1