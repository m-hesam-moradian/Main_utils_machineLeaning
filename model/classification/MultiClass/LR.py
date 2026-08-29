import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "D4_Data" 

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Split into train/test (80/20, shuffle=False to match K-Fold) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --- Train Logistic Regression with StandardScaler ---
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=300, random_state=42)
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
print("[LR] Logistic Regression Accuracy (D4)")
print("-----------------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")

# --- Get predicted probabilities ---
y_pred_proba = model.predict_proba(X)
classes = model.named_steps["logisticregression"].classes_

# Convert predicted probabilities to DataFrame
proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in classes]
)

# Combine results
df_all = pd.concat([
    pd.DataFrame({"y_real": y.values, "y_pred": y_pred_all}),
    proba_df
], axis=1)

# Export to .npt files
df_all.to_csv(r"data/Data_err.npt", sep="\t", index=False, header=False)
print("Saved predictions to data/model1.npt & data/Data_err.npt")
