import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score

# --- Load reordered data for ADAC (after K-Fold) ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_ADAC"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]

# Prepare the features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Use last 20% as test set to match K-Fold logic ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Initialize ADAC model ---
model = AdaBoostClassifier(n_estimators=800, learning_rate=0.5)

# Train the model
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Metrics (Classification) ---
mid = len(y_test) // 2
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test[:mid], y_pred_test[:mid]),
    ("Test-Value", y_test[mid:], y_pred_test[mid:])
]

# Calculate Accuracy, F1 Score, and Precision for each set
df_metrics = pd.DataFrame([{
    "Set": s,
    "Accuracy": accuracy_score(y_t, y_p),
    "F1 Score": f1_score(y_t, y_p, average='weighted'),
    "Precision": precision_score(y_t, y_p, average='weighted', zero_division=0)
} for s, y_t, y_p in sets])

print("\nADAC Metrics:")
print(df_metrics)

# --- Get Probabilities (Multi-class) ---
y_pred_proba = model.predict_proba(X)

# Create probability columns dynamically based on classes
proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in model.classes_]
)

# Combine with true and predicted labels
df_all = pd.concat([
    pd.DataFrame({"y_real": y, "y_pred": y_pred_all}),
    proba_df
], axis=1)

print("\n📊 Sample of ADAC predictions with probabilities:")
print(df_all.head())

# --- Export to clipboard (No header, No index) ---
df_all.to_clipboard(index=False, header=False)
print("✅ ADAC Results with probabilities copied to clipboard.")