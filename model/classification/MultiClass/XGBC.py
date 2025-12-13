import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier   # <-- changed import

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_GPC"  # renamed to reflect XGBC

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Split into train/test (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train Extreme Gradient Boosting Classifier ---
model = XGBClassifier(
    # use_label_encoder=False,   # suppress label encoder warning
    # eval_metric="logloss",     # evaluation metric
    n_estimators=2000,          # number of boosting rounds
    # max_depth=1000,               # tree depth
    # learning_rate=0.01,         # step size shrinkage
    subsample=0.0018,             # subsample ratio
    # colsample_bytree=0.8,      # feature subsample ratio
    # random_state=42
)

model.fit(X_train, y_train)

# --- Predictions ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X)  # full data

# --- Accuracy metrics ---
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
acc_all = accuracy_score(y, y_pred_all)

# --- Print neatly ---
print("✅ Accuracy Results (XGBoost Classifier)")
print("----------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")

# --- Get predicted probabilities ---
y_pred_proba = model.predict_proba(X)

# Convert predicted probabilities to a DataFrame with one column per class
proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in model.classes_]
)

# Combine with true and predicted labels
df_all = pd.concat([
    pd.DataFrame({"y_real": y, "y_pred": y_pred_all}),
    proba_df
], axis=1)
df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# --- Optional: view first few rows ---
print("\n📊 Sample of overall predictions:")
print(df_all.head())

# --- Optional: export to clipboard or Excel ---
df_all.to_clipboard(index=False, header=False)