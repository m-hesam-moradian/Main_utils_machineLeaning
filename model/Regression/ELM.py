import pandas as pd
import numpy as np
import os
import win32com.client
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ================== Excel Helpers ==================
def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

def open_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[*] Opened Excel file:", filepath)
    except Exception:
        pass

# ================== ELM Regressor Definition ==================
class ELMRegressor:
    def __init__(self, n_hidden=120, alpha=0.1, activation='sigmoid', random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.activation = activation
        self.random_state = random_state

    def _act(self, x):
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        return x

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.w = rng.normal(0, 1, (n_features, self.n_hidden))
        self.b = rng.normal(0, 1, (1, self.n_hidden))
        H = self._act(np.dot(X, self.w) + self.b)
        HtH = np.dot(H.T, H) + self.alpha * np.eye(self.n_hidden)
        Hty = np.dot(H.T, y)
        self.beta = np.linalg.solve(HtH, Hty)
        return self

    def predict(self, X):
        H = self._act(np.dot(X, self.w) + self.b)
        return np.dot(H, self.beta)

# ================== Load Dataset ==================
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

df = pd.read_excel(excel_path, sheet_name="Data_after_KFold_ELM")
target_column = "Reliability_Score"
X = df.drop(columns=[target_column]).values
y = df[target_column].values

# Split data (80/20 train/test split, shuffle=False)
n_train = int(len(df) * 0.8)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
X_all_sc = scaler.transform(X)

# Model Training
elm = ELMRegressor(n_hidden=120, alpha=0.1, activation='sigmoid', random_state=42)
elm.fit(X_train_sc, y_train)

# Predictions
y_pred_train = elm.predict(X_train_sc)
y_pred_test = elm.predict(X_test_sc)
y_pred_all = elm.predict(X_all_sc)

# Metric Calculations
def calc_metrics(y_true, y_hat):
    r2 = r2_score(y_true, y_hat)
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    mape = np.mean(np.abs((y_true - y_hat) / y_true)) * 100.0
    ratio = y_hat / y_true
    mv = np.mean(ratio)
    cov = np.std(ratio, ddof=1) / mv
    return {"R2": r2, "RMSE": rmse, "MAPE (%)": mape, "MV": mv, "COV": cov}

df_metrics = pd.DataFrame([
    {"Partition": "Train", **calc_metrics(y_train, y_pred_train)},
    {"Partition": "Test", **calc_metrics(y_test, y_pred_test)},
    {"Partition": "All", **calc_metrics(y, y_pred_all)}
])

print("=== ELM Regression Metrics ===")
print(df_metrics.to_string(index=False))
