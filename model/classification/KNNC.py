import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score

# --- Columns to drop ---
# COLUMNS_TO_DROP = ['workload_type', 'energy_source', 'security_level', 'pqc_enabled']

# --- Load reordered data for KNNC (after K-Fold) ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_KNNC"  # Changed to "KNNC" sheet name

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]

# Drop the specified columns from the dataset
# df = df.drop(columns=COLUMNS_TO_DROP)

# Prepare the features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Use last 20% as test set to match K-Fold logic ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Initialize KNNC model ---
# n_neighbors=5 is the standard default for KNN
model = KNeighborsClassifier(n_neighbors=17)

# Train the model
model.fit(X_train, y_train)

# Predictions
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

print(df_metrics)

# --- Output predictions ---
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# --- Export to clipboard ---
df_all.to_clipboard(index=False, header=False)
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)