import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_RNN"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Separate features and target ---
target_column = df.columns[-1]
X = df.drop(columns=[target_column]).values
y = df[target_column].values

# --- Standardize Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split into train/test (80/20, shuffle=False to match K-Fold) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# --- Train RNN (MLP) Classifier ---
model = MLPClassifier(
    hidden_layer_sizes=(16,),
    max_iter=100,
    alpha=5.0,
    random_state=44
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

print("[RNN] Neural Network Accuracy")
print("---------------------------------")
print(f"Overall Accuracy  : {acc_all:.4f}")
print(f"Training Accuracy : {acc_train:.4f}")
print(f"Testing Accuracy  : {acc_test:.4f}")

# --- Get predicted probabilities ---
y_pred_proba = model.predict_proba(X_scaled)

proba_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"Prob_Class_{cls}" for cls in model.classes_]
)

# Combine results
df_all = pd.concat([
    pd.DataFrame({"y_real": y, "y_pred": y_pred_all}),
    proba_df
], axis=1)

# Export to .npt files
df_all.to_csv(r"data/model_RNN.npt", sep="\t", index=False, header=False)
df_all.to_csv(r"data/model1.npt", sep="\t", index=False, header=False)
df_all.to_csv(r"data/Data_err.npt", sep="\t", index=False, header=False)
print("Saved predictions to data/model_RNN.npt, model1.npt, Data_err.npt")
