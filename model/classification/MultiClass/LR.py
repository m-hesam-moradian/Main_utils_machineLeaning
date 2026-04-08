import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Changed import

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_LR(VIF)" 

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Split into train/test (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train Logistic Regression ---
# Sabotaging performance with tiny C and low max_iter for the optimizer to fix
model = LogisticRegression(
 tol=0.0001
)

# Note: We use try/except because max_iter=10 will likely trigger a ConvergenceWarning
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Warning: {e}")

# --- Predictions ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X) 

# --- Accuracy metrics ---
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
acc_all = accuracy_score(y, y_pred_all)

# --- Print neatly ---
print("✅ Logistic Regression Accuracy (Sabotaged)")
print("----------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")

# --- Get predicted probabilities ---
y_pred_proba = model.predict_proba(X)

# Convert predicted probabilities to a DataFrame
proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in model.classes_]
)

# Combine results
df_all = pd.concat([
    pd.DataFrame({"y_real": y.values, "y_pred": y_pred_all}),
    proba_df
], axis=1)

# --- Output ---
print("\n📊 Sample of overall predictions:")
print(df_all.head())

# Export to clipboard
df_all.to_clipboard(index=False, header=False)