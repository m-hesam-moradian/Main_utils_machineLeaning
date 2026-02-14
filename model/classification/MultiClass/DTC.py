import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_DTC" 

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]

# FIXED: Use .iloc for clean indexing to avoid the KeyError
X = df.iloc[:, :-1]  
y = df[target_column]

# --- Split into train/test (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train DTC model ---
model = DecisionTreeClassifier(
        max_depth=7,  # If accuracy is still 1.0, lower this to 2 or 3
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

# --- Print neatly ---
# FIXED: Use X.columns to get the count since 'noisy_features' wasn't defined
print(f"✅ DTC Results (using {len(X.columns)} features)")
print("----------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")

# --- Output Sample ---
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
print("\n📊 Sample of DTC predictions:")
print(df_all.head())
df_all.to_clipboard(index=False)  # Copy the original DataFrame to clipboard for Excel