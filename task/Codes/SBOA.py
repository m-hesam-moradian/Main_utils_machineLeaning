import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import QuantileRegressor
import pandas as pd
# Load dataset
import pandas as pd
# Load sample dataset for demonstration
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.22-Data.xlsx"
sheet_name = "Z-score"
target_column = "Renewable Availability Index"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Secretary Bird Optimization Algorithm (SBOA)
def sboa_optimize(objective_func, lb, ub, N=30, T=100):
    D = len(lb)
    population = lb + np.random.rand(N, D) * (ub - lb)
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]
    worst_idx = np.argmax(fitness)
    worst_params = population[worst_idx].copy()

    for t in range(T):
        for i in range(N):
            r1 = np.random.rand(D)
            new_pos = population[i] + r1 * (population[i] - best_params)
            new_pos = np.clip(new_pos, lb, ub)
            new_fit = objective_func(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit

            r2 = np.random.rand(D)
            new_pos = population[i] + r2 * (population[i] - worst_params)
            new_pos = np.clip(new_pos, lb, ub)
            new_fit = objective_func(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit

        worst_idx = np.argmax(fitness)
        worst_params = population[worst_idx].copy()

    return best_params, best_score

# Objective: Elastic Net Regression (ENR)
def objective_enr(params):
    alpha = params[0]
    l1_ratio = params[1]
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error").mean()
    return -score

# Objective: Extreme Gradient Boosting Regression (XGBR)
def objective_xgbr(params):
    n_est = int(params[0])
    lr = params[1]
    md = int(params[2])
    model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error").mean()
    return -score

# Objective: Quantile Regression (QR)
def objective_qr(params):
    alpha = params[0]
    quantile = params[1]
    model = QuantileRegressor(alpha=alpha, quantile=quantile, solver="highs", max_iter=10000)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error").mean()
    return -score

# Hyperparameter bounds
lb_enr = np.array([0.001, 0.0])
ub_enr = np.array([1.0, 1.0])

lb_xgbr = np.array([50, 0.01, 3])
ub_xgbr = np.array([200, 0.3, 10])

lb_qr = np.array([0.001, 0.1])
ub_qr = np.array([1.0, 0.9])

# Optimize ENR
print("Optimizing Elastic Net Regression (ENR) with SBOA...")
best_params_enr, best_score_enr = sboa_optimize(objective_enr, lb_enr, ub_enr, N=20, T=50)
print(f"Best ENR params: alpha={best_params_enr[0]:.4f}, l1_ratio={best_params_enr[1]:.4f}")
print(f"Best CV MSE: {best_score_enr:.4f}")
enr_final = ElasticNet(alpha=best_params_enr[0], l1_ratio=best_params_enr[1], random_state=42, max_iter=10000)
enr_final.fit(X_train, y_train)
y_pred_enr = enr_final.predict(X_test)
print(f"Test MSE for ENR: {mean_squared_error(y_test, y_pred_enr):.4f}\n")

# Optimize XGBR
print("Optimizing Extreme Gradient Boosting Regression (XGBR) with SBOA...")
best_params_xgbr, best_score_xgbr = sboa_optimize(objective_xgbr, lb_xgbr, ub_xgbr, N=20, T=50)
print(f"Best XGBR params: n_estimators={int(best_params_xgbr[0])}, learning_rate={best_params_xgbr[1]:.4f}, max_depth={int(best_params_xgbr[2])}")
print(f"Best CV MSE: {best_score_xgbr:.4f}")
xgbr_final = GradientBoostingRegressor(n_estimators=int(best_params_xgbr[0]), learning_rate=best_params_xgbr[1], max_depth=int(best_params_xgbr[2]), random_state=42)
xgbr_final.fit(X_train, y_train)
y_pred_xgbr = xgbr_final.predict(X_test)
print(f"Test MSE for XGBR: {mean_squared_error(y_test, y_pred_xgbr):.4f}\n")

# Optimize QR
print("Optimizing Quantile Regression (QR) with SBOA...")
best_params_qr, best_score_qr = sboa_optimize(objective_qr, lb_qr, ub_qr, N=20, T=50)
print(f"Best QR params: alpha={best_params_qr[0]:.4f}, quantile={best_params_qr[1]:.2f}")
print(f"Best CV MSE: {best_score_qr:.4f}")
qr_final = QuantileRegressor(alpha=best_params_qr[0], quantile=best_params_qr[1], solver="highs", max_iter=10000)
qr_final.fit(X_train, y_train)
y_pred_qr = qr_final.predict(X_test)
print(f"Test MSE for QR: {mean_squared_error(y_test, y_pred_qr):.4f}")