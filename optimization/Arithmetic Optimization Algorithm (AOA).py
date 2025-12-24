import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
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


# Arithmetic Optimization Algorithm (AOA) implementation
def aoa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Arithmetic Optimization Algorithm (AOA) for minimizing an objective function.
    Based on basic arithmetic operations (+, -, *, /) applied in a search process.
    """
    D = len(lb)
    # Initialize population
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        # Adaptive parameter: decreases over time (exploration → exploitation)
        alpha = 0.5 * (1 - t / T) + 0.1  # Range: ~0.6 → 0.1

        for i in range(N):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            r3 = np.random.rand(D)
            r4 = np.random.rand(D)

            # Arithmetic operators applied to current best
            if r1 < 0.5:
                # Addition/Subtraction
                if r2 < 0.5:
                    new_pos = best_params + alpha * r3 * (ub - lb) * (r4 - 0.5)
                else:
                    new_pos = best_params - alpha * r3 * (ub - lb) * (r4 - 0.5)
            else:
                # Multiplication/Division
                if r2 < 0.5:
                    new_pos = best_params * (1 + alpha * r3 * (r4 - 0.5))
                else:
                    new_pos = best_params / (1 + alpha * r3 * (r4 - 0.5) + 1e-10)  # avoid div by zero

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)

            # Update if better
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t}: Best params = {best_params}, Best score = {best_score:.4f}")

        # Occasional diversification (random restart for worst)
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.05:
            population[worst_idx] = lb + np.random.rand(D) * (ub - lb)
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


# Objective function for Decision Trees (DT)
def objective_dt(params):
    """
    Objective for DecisionTreeRegressor.
    Params: [max_depth, min_samples_split, min_samples_leaf]
    """
    md = int(params[0])
    mss = int(params[1])
    msl = int(params[2])

    model = DecisionTreeRegressor(
        max_depth=md, min_samples_split=mss, min_samples_leaf=msl, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Hyperparameter bounds for SVR
lb_svr = np.array([1, 0.001, 0])
ub_svr = np.array([1000, 1, 2])

# Hyperparameter bounds for DT
lb_dt = np.array([3, 2, 1])
ub_dt = np.array([20, 20, 10])


# Optimize SVR using AOA
print("Optimizing SVR with AOA...")
best_params_svr, best_score_svr = aoa_optimize(
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

# Optimize DT using AOA
print("Optimizing DT with AOA...")
best_params_dt, best_score_dt = aoa_optimize(
    objective_dt, lb_dt, ub_dt, N=20, T=50
)
print(
    f"Best DT params: max_depth={int(best_params_dt[0])}, min_samples_split={int(best_params_dt[1])}, min_samples_leaf={int(best_params_dt[2])}"
)
print(f"Best CV MSE: {best_score_dt:.4f}")

# Train final DT model and evaluate on test set
dt_final = DecisionTreeRegressor(
    max_depth=int(best_params_dt[0]),
    min_samples_split=int(best_params_dt[1]),
    min_samples_leaf=int(best_params_dt[2]),
    random_state=42
)
dt_final.fit(X_train, y_train)
y_pred_dt = dt_final.predict(X_test)
test_mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f"Test MSE for DT: {test_mse_dt:.4f}")