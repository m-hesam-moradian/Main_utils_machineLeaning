import pandas as pd
import numpy as np
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
# Using your parameters from the first script
model = KNeighborsClassifier(n_neighbors=17)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Get Predicted Probabilities ---
# This extracts the probability for each class (0, 1, 2)
y_pred_proba = model.predict_proba(X)

# Convert predicted probabilities to a DataFrame with one column per class
# Using model.classes_ ensures the columns match the data order
proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{int(c)}" for c in model.classes_]
)

# --- Combine Real, Predicted, and Probabilities ---
# Order: y_real, y_pred, Prob_Class_0, Prob_Class_1, Prob_Class_2
df_all = pd.concat([
    pd.DataFrame({"y_real": y, "y_pred": y_pred_all}),
    proba_df
], axis=1)

# --- Metrics (Optional Check) ---
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

print("Metrics Summary:")
print(df_metrics)
print("\nPreview of Output Data:")
print(df_all.head())

# --- Export to clipboard (Format: y_real, y_pred, prob0, prob1, prob2) ---
# Using header=False as requested by your output format
df_all.to_clipboard(index=False, header=False)
print("\n✅ Output (Real, Pred, Probabilities) copied to clipboard.")