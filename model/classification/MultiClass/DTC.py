import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Changed to DTC

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_DTC" 

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Split into train/test (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train Decision Tree Classifier ---
# max_depth=2 keeps the tree small and prevents overfitting
model = DecisionTreeClassifier(
    criterion="gini",          # or "entropy"
    max_depth=3,               # depth limit
    min_samples_split=20,      # needs 10 samples to split a node
    min_samples_leaf=10,        # each leaf must have at least 5 samples
    max_features="sqrt",       # looks at sqrt(n_features) at each split
    class_weight="balanced",   # helps if your labels are 90/10 split
    random_state=42
)






model.fit(X_train, y_train)

# --- Predictions ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X)

# --- Accuracy metrics ---
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
acc_all = accuracy_score(y, y_pred_all)

# --- Print results ---
print("✅ Decision Tree Accuracy Results")
print("----------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")

# --- Get predicted probabilities ---
y_pred_proba = model.predict_proba(X)
proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in model.classes_]
)

# Combine results
df_all = pd.concat([
    pd.DataFrame({"y_real": y.values, "y_pred": y_pred_all}),
    proba_df
], axis=1)

# --- Export ---
# Note: I added index=True/header=True usually preferred for Excel/Clipboard
df_all.to_clipboard(index=False, header=True)
print("\n📋 Results copied to clipboard!")