import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

# ==========================================
# PART 1: K-Nearest Neighbors (KNN) Classifier
# ==========================================
print("Running KNN Classifier...")

# Load data
sheet_name_knn = "Data_after_KFold_ADAC"
df_knn = pd.read_excel(excel_path, sheet_name=sheet_name_knn)

# Prepare features and target
target_column = df_knn.columns[-1]
X = df_knn.drop(columns=[target_column])
y = df_knn[target_column]

# --- Use last 20% as test set to match K-Fold logic (Sequential Split) ---
split_idx = int(len(df_knn) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Initialize KNNC model ---
model = KNeighborsClassifier(n_neighbors=684)
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

df_metrics = pd.DataFrame([{
    "Set": s,
    "Accuracy": accuracy_score(y_t, y_p),
    "F1 Score": f1_score(y_t, y_p, average='weighted'),
    "Precision": precision_score(y_t, y_p, average='weighted', zero_division=0)
} for s, y_t, y_p in sets])

print("\nKNN Metrics:")
print(df_metrics)

# --- Get Probabilities & Export (Updated to match GPC style) ---
y_pred_proba = model.predict_proba(X)

# Create probability columns
proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in model.classes_]
)

# Combine with true and predicted labels
df_all = pd.concat([
    pd.DataFrame({"y_real": y, "y_pred": y_pred_all}),
    proba_df
], axis=1)

print("\n📊 Sample of KNN predictions with probabilities:")
print(df_all.head())

# Export to clipboard (No header, no index)
df_all.to_clipboard(index=False, header=False)
print("✅ KNN Results copied to clipboard.")

