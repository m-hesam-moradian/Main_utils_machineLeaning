import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score

# --- Load data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "test"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]

# --- Features & Target ---
X = df.drop(columns=[target_column])
y = df[target_column]

# ✅ STEP 1: Convert IELTS bands → class labels
# Example: 1 → 0, 1.5 → 1, 2 → 2, ..., 5 → 8
y_class = ((y - 1) * 2).astype(int)

# --- Train/Test Split ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_class[:split_idx], y_class[split_idx:]

# --- Model ---
model = XGBClassifier(
    n_estimators=300,
    max_depth=2,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# --- Train ---
model.fit(X_train, y_train)

# --- Predictions (class form) ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ✅ STEP 2: Convert back to IELTS bands
def to_ielts(x):
    return (x / 2) + 1

y_pred_all_band = to_ielts(y_pred_all)
y_pred_train_band = to_ielts(y_pred_train)
y_pred_test_band = to_ielts(y_pred_test)

y_test_band = to_ielts(y_test)

# --- Metrics (Classification) ---
mid = len(y_test) // 2
sets = [
    ("All", y_class, y_pred_all),
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

# --- Output predictions (IELTS format) ---
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all_band})
df_train = pd.DataFrame({"y_real": y[:split_idx], "y_pred": y_pred_train_band})
df_test = pd.DataFrame({"y_real": y[split_idx:], "y_pred": y_pred_test_band})

# --- Export ---
df_all.to_clipboard(index=False, header=False)