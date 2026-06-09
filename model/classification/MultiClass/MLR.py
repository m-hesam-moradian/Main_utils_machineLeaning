import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_MLR"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Split into train/test (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Feature Scaling (IMPORTANT for MLR) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# --- Train Multinomial Logistic Regression ---
model = LogisticRegression(
    multi_class='multinomial',
    # solver='lbfgs',
    max_iter=10,
    C=0.043,
    random_state=42
)

model.fit(X_train, y_train)

# --- Predictions ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X_scaled)

# --- Accuracy metrics ---
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
acc_all = accuracy_score(y, y_pred_all)

# --- Print neatly ---
print("✅ Accuracy Results")
print("----------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")

# --- Get predicted probabilities ---
y_pred_proba = model.predict_proba(X_scaled)

# Convert predicted probabilities to a DataFrame with one column per class
proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in model.classes_],
    index=df.index
)

# Combine with true and predicted labels
df_all = pd.concat([
    pd.DataFrame({"y_real": y, "y_pred": y_pred_all}, index=df.index),
    proba_df
], axis=1)

df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# --- Optional: view first few rows ---
print("\n📊 Sample of overall predictions:")
print(df_all.head())

# --- Optional: export ---
df_all.to_clipboard(index=False, header=False)
# df_all.to_excel("mlr_predictions.xlsx", index=False)