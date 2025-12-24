import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Load sample dataset for demonstration (California Housing for regression)
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Chimp Optimization Algorithm (ChOA) implementation
def choa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Chimp Optimization Algorithm (ChOA) for minimizing an objective function.
    Inspired by the social hierarchy and hunting behavior of chimpanzees (alpha, beta, delta, and chaotic exploration).
    """
    D = len(lb)
    # Initialize population (chimps)
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    # Initialize alpha, beta, delta
    alpha_pos = best_params.copy()
    beta_pos = population[np.argsort(fitness)[1]].copy()
    delta_pos = population[np.argsort(fitness)[2]].copy()

    for t in range(1, T + 1):
        # Adaptive parameters
        a = 2 * (1 - t / T)
        r1 = np.random.rand(N, D)
        r2 = np.random.rand(N, D)
        chaotic_factor = np.random.rand(N, D) * 2 * np.pi  # Chaotic behavior

        for i in range(N):
            # Update position based on alpha, beta, delta
            if np.random.rand() < 0.5:
                # Exploitation phase (follow leaders)
                distance_alpha = abs(r1[i] * alpha_pos - population[i])
                distance_beta = abs(r2[i] * beta_pos - population[i])
                new_pos = alpha_pos - a * r1[i] * distance_alpha + chaotic_factor[i] * (beta_pos - population[i])
            else:
                # Exploration phase (chaotic jump)
                distance_delta = abs(r1[i] * delta_pos - population[i])
                new_pos = delta_pos - a * r2[i] * distance_delta + chaotic_factor[i] * (alpha_pos - population[i])

            # Boundary handling
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)

            # Greedy update
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit

                # Update leaders
                if new_fit < best_score:
                    alpha_pos = new_pos.copy()
                    best_score = new_fit
                    best_params = new_pos.copy()
                    print(f"Iteration {t}: Best params = {best_params}, Best score = {best_score:.4f}")

                # Update beta/delta (simplified)
                sorted_idx = np.argsort(fitness)
                beta_pos = population[sorted_idx[1]].copy()
                delta_pos = population[sorted_idx[2]].copy()

        # Chaotic perturbation at the end of iteration (diversification)
        if t % 10 == 0:
            worst_idx = np.argmax(fitness)
            population[worst_idx] += (ub - lb) * np.random.randn(D) * 0.1
            population[worst_idx] = np.clip(population[worst_idx], lb, ub)
            fitness[worst_idx] = objective_func(population[worst_idx])

    return best_params, best_score


# Objective function for Support Vector Regression (SVR)
def objective_svr(params):
    """
    Objective for SVR.
    Params: [C, gamma, kernel_index (0: rbf, 1: poly, 2: linear)]
    """
    C = params[0]
    gamma = params[1]
    kernel_idx = int(params[2])
    kernel = ['rbf', 'poly', 'linear'][kernel_idx]

    model = SVR(
        C=C, gamma=gamma, kernel=kernel
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score  # Minimize MSE


# Objective function for Extreme Gradient Boosting Regression (XGBR)
def objective_xgbr(params):
    """
    Objective for XGBRegressor.
    Params: [max_depth, learning_rate, n_estimators, subsample]
    """
    md = int(params[0])
    lr = params[1]
    ne = int(params[2])
    ss = params[3]

    model = XGBRegressor(
        max_depth=md, learning_rate=lr, n_estimators=ne, subsample=ss, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Hyperparameter bounds for SVR
lb_svr = np.array([1, 0.001, 0])
ub_svr = np.array([1000, 1, 2])

# Hyperparameter bounds for XGBR
lb_xgbr = np.array([3, 0.01, 50, 0.5])
ub_xgbr = np.array([15, 0.3, 500, 1.0])

# Optimize SVR using ChOA
print("Optimizing SVR with ChOA...")
best_params_svr, best_score_svr = choa_optimize(
    objective_svr, lb_svr, ub_svr, N=20, T=50
)
print(
    f"Best SVR params: C={best_params_svr[0]:.4f}, gamma={best_params_svr[1]:.4f}, kernel={['rbf', 'poly', 'linear'][int(best_params_svr[2])]}"
)
print(f"Best CV MSE: {best_score_svr:.4f}")

# Train final SVR model and evaluate on test set
kernel = ['rbf', 'poly', 'linear'][int(best_params_svr[2])]
svr_final = SVR(
    C=best_params_svr[0],
    gamma=best_params_svr[1],
    kernel=kernel
)
svr_final.fit(X_train, y_train)
y_pred_svr = svr_final.predict(X_test)
test_mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"Test MSE for SVR: {test_mse_svr:.4f}\n")

# Optimize XGBR using ChOA
print("Optimizing XGBR with ChOA...")
best_params_xgbr, best_score_xgbr = choa_optimize(
    objective_xgbr, lb_xgbr, ub_xgbr, N=20, T=50
)
print(
    f"Best XGBR params: max_depth={int(best_params_xgbr[0])}, learning_rate={best_params_xgbr[1]:.4f}, n_estimators={int(best_params_xgbr[2])}, subsample={best_params_xgbr[3]:.4f}"
)
print(f"Best CV MSE: {best_score_xgbr:.4f}")

# Train final XGBR model and evaluate on test set
xgbr_final = XGBRegressor(
    max_depth=int(best_params_xgbr[0]),
    learning_rate=best_params_xgbr[1],
    n_estimators=int(best_params_xgbr[2]),
    subsample=best_params_xgbr[3],
    random_state=42
)
xgbr_final.fit(X_train, y_train)
y_pred_xgbr = xgbr_final.predict(X_test)
test_mse_xgbr = mean_squared_error(y_test, y_pred_xgbr)
print(f"Test MSE for XGBR: {test_mse_xgbr:.4f}")