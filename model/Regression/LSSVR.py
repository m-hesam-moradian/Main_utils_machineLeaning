import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# --- Define the custom LSSVR class for Scikit-Learn Ecosystem ---
class LSSVR(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=None, coef0=1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def _get_kernel_params(self):
        params = {}
        if self.kernel == 'poly':
            params['degree'] = self.degree
            params['coef0'] = self.coef0
        if self.gamma is not None:
            params['gamma'] = self.gamma
        return params

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.X_train_ = X
        self.y_train_ = y
        
        n_samples = X.shape[0]
        kernel_params = self._get_kernel_params()
        K = pairwise_kernels(X, X, metric=self.kernel, **kernel_params)
        
        # Build the LSSVM linear system matrix
        H = np.zeros((n_samples + 1, n_samples + 1))
        H[0, 1:] = 1.0
        H[1:, 0] = 1.0
        H[1:, 1:] = K + np.eye(n_samples) / self.C
        
        RHS = np.zeros(n_samples + 1)
        RHS[1:] = y
        
        try:
            solution = np.linalg.solve(H, RHS)
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(H).dot(RHS)
            
        self.bias_ = solution[0]
        self.alphas_ = solution[1:]
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_train_', 'y_train_', 'bias_', 'alphas_'])
        X = check_array(X)
        kernel_params = self._get_kernel_params()
        K_test = pairwise_kernels(X, self.X_train_, metric=self.kernel, **kernel_params)
        return np.dot(K_test, self.alphas_) + self.bias_


# --- Load reordered data for LSSVR (after K-Fold) ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_After_ANOVA"   # keep same sheet unless you renamed it

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column1 = df.columns[-1]
target_column2 = df.columns[-2]

X = df.drop(columns=[target_column1, target_column2])
y = df[target_column2]

# --- Use last 20% as test set to match K-Fold logic ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Define and train LSSVR model ---
# NOTE: epsilon parameter is dropped since LSSVR ignores epsilon-insensitive losses.
model = LSSVR(
    kernel="poly",          # Switched to poly to utilize the degree parameter
    C=19.7848895,           # Regularization parameter
    degree=5,               # Polynomial degree parameter
    coef0=1                 # Independent term in kernel function
)

model.fit(X_train.to_numpy(), y_train.to_numpy())

# --- Predictions ---
# Converting to numpy arrays during prediction to ensure clean matrix algebra
y_pred_all = model.predict(X.to_numpy())
y_pred_train = model.predict(X_train.to_numpy())
y_pred_test = model.predict(X_test.to_numpy())

# --- Metrics ---
mid = len(y_test) // 2
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test[:mid], y_pred_test[:mid]),
    ("Test-Value", y_test[mid:], y_pred_test[mid:])
]

df_metrics = pd.DataFrame([{
    "Set": s,
    "MAE": mean_absolute_error(y_t, y_p),
    "RMSE": mean_squared_error(y_t, y_p) ** 0.5,
    "R2": r2_score(y_t, y_p)
} for s, y_t, y_p in sets])

print(df_metrics)

# --- Output predictions ---
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# --- Export to clipboard ---
df_all.to_clipboard(index=False, header=False)
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)