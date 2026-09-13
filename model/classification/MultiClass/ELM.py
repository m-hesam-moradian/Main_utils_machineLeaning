import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Custom ELM Implementation ---
class ELMClassifier:
    def __init__(self, n_hidden=150, alpha=0.5, random_state=44):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.W = rng.normal(size=(X.shape[1], self.n_hidden))
        self.b = rng.normal(size=(self.n_hidden,))
        H = 1.0 / (1.0 + np.exp(- (X @ self.W + self.b)))
        num_classes = len(np.unique(y))
        self.classes_ = np.unique(y)
        Y_oh = np.eye(num_classes)[y]
        HtH = H.T @ H + self.alpha * np.eye(self.n_hidden)
        self.beta = np.linalg.solve(HtH, H.T @ Y_oh)
        return self

    def predict(self, X):
        H = 1.0 / (1.0 + np.exp(- (X @ self.W + self.b)))
        scores = H @ self.beta
        return np.argmax(scores, axis=1)

    def predict_proba(self, X):
        H = 1.0 / (1.0 + np.exp(- (X @ self.W + self.b)))
        scores = H @ self.beta
        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

# --- Load Excel file ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_ELM"

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

# --- Train ELM Model ---
model = ELMClassifier(n_hidden=150, alpha=0.5, random_state=44)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X_scaled)

# --- Accuracy metrics ---
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
acc_all = accuracy_score(y, y_pred_all)

print("[ELM] Extreme Learning Machine Accuracy")
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
df_all.to_csv(r"data/model_ELM.npt", sep="\t", index=False, header=False)
print("Saved predictions to data/model_ELM.npt")
