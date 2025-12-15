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


# Ladybug Beetle Optimization (LBO) implementation
def lbo_optimize(objective_func, lb, ub, N0=30, k_max=100, beta=10, mu_m=0.2):
    """
    Ladybug Beetle Optimization for minimizing an objective function.
    Based on ladybug winter aggregation behavior: exploration via mutation, exploitation via position updates with roulette-wheel selection and cost ratios.
    """
    D = len(lb)
    N_k = N0
    # Initialize population
    population = lb + np.random.rand(N0, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]
    f_worst = np.max(fitness)  # Historical worst

    for k in range(1, k_max + 1):
        # Sort population by fitness (ascending for minimization)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(N_k):
            # Compute cost ratios C
            sum_f = np.sum(fitness)
            C = fitness[i] / sum_f if sum_f != 0 else 1.0 / N_k

            # Mutation for exploration
            if np.random.rand() < 0.1:  # Mutation probability (adjustable)
                n_m = int(np.round(D * mu_m))
                mut_idx = np.random.choice(D, n_m, replace=False)
                new_pos = population[i].copy()
                new_pos[mut_idx] = lb[mut_idx] + np.random.rand(n_m) * (ub[mut_idx] - lb[mut_idx])
            else:
                # Roulette-wheel selection for j
                P = np.exp(-beta * fitness / f_worst)
                P = P / np.sum(P)
                j = np.random.choice(range(N_k), p=P)

                # Position update (simplified from description)
                rand1 = np.random.rand(D)
                rand2 = np.random.rand(D)
                rand3 = np.random.rand(D)
                new_pos = population[i] + rand1 * (population[j] - population[i]) + rand2 * (population[j] - population[0]) + rand3 * C * (population[i] - best_params)  # population[0] is best after sort

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)

            # Update if better
            if new_fit < fitness[i]:
                new_population[i] = new_pos
                new_fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {k}: Best params = {best_params}, Best score = {best_score:.4f}")

        population = new_population
        fitness = new_fitness

        # Update historical worst
        f_worst = max(f_worst, np.max(fitness))

        # Population annihilation (reduction)
        N_next = max(int(0.25 * N0), int(np.round(N_k - np.random.rand() * N_k * (k / k_max))))
        if N_next < N_k:
            # Keep top N_next
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx[:N_next]]
            fitness = fitness[sorted_idx[:N_next]]
            N_k = N_next

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

# Optimize SVR using LBO
print("Optimizing SVR with LBO...")
best_params_svr, best_score_svr = lbo_optimize(
    objective_svr, lb_svr, ub_svr, N0=30, k_max=50, beta=10, mu_m=0.2
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

# Optimize XGBR using LBO
print("Optimizing XGBR with LBO...")
best_params_xgbr, best_score_xgbr = lbo_optimize(
    objective_xgbr, lb_xgbr, ub_xgbr, N0=30, k_max=50, beta=10, mu_m=0.2
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