import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_DTC"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# CATBOOST CLASSIFIER (CATC)
# =====================================================

model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function="MultiClass" if y.nunique() > 2 else "Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0  # set to 50 if you want logs
)

# --- Train ---
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X)

# --- Accuracy ---
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
acc_all = accuracy_score(y, y_pred_all)

print("✅ Accuracy Results (CATC)")
print("----------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")

# --- Predicted probabilities ---
y_pred_proba = model.predict_proba(X)

proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in model.classes_]
)

df_all = pd.concat([
    pd.DataFrame({"y_real": y, "y_pred": y_pred_all}),
    proba_df
], axis=1)

print("\n📊 Sample of overall predictions:")
print(df_all.head())

# --- Export ---
df_all.to_clipboard(index=False, header=False)
