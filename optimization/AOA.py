import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load sample dataset for demonstration
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Addax Optimization Algorithm (AOA) implementation
def aoa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Addax Optimization Algorithm for minimizing an objective function.

    Parameters:
    - objective_func: function that takes a 1D array of parameters and returns a scalar value to minimize.
    - lb: lower bounds (array of length D)
    - ub: upper bounds (array of length D)
    - N: population size
    - T: number of iterations

    Returns:
    - best_params: optimized parameters
    - best_score: best objective value
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
        for i in range(N):
            # Phase 1: Foraging (Exploration)
            # Find candidates with better fitness
            better_indices = [k for k in range(N) if fitness[k] < fitness[i] and k != i]
            if better_indices:
                # Select random better position
                s_idx = np.random.choice(better_indices)
                S = population[s_idx]

                # Update position
                r = np.random.rand(D)
                I = np.random.choice([1, 2], size=D)
                new_pos = population[i] + r * (S - I * population[i])

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

            # Phase 2: Digging (Exploitation)
            r = np.random.rand(D)
            perturbation = (1 - 2 * r) * (ub - lb) / t
            new_pos = population[i] + perturbation

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

    return best_params, best_score


# Objective function for Stochastic Gradient Boosting (SGB)
def objective_sgb(params):
    """
    Objective for GradientBoostingRegressor (SGB).
    Params: [n_estimators, learning_rate, max_depth]
    """
    n_est = int(params[0])
    lr = params[1]
    md = int(params[2])

    model = GradientBoostingRegressor(
        n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=42
    )

    # Use cross-validation score (neg_mean_squared_error)
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score  # Minimize MSE (convert to positive MSE)


# Objective function for LightGBM Regressor (LGBR)
def objective_lgbr(params):
    """
    Objective for LGBMRegressor.
    Params: [num_leaves, learning_rate, n_estimators]
    """
    num_leaves = int(params[0])
    lr = params[1]
    n_est = int(params[2])

    model = lgb.LGBMRegressor(
        num_leaves=num_leaves,
        learning_rate=lr,
        n_estimators=n_est,
        random_state=42,
        verbosity=-1,  # Suppress warnings
    )

    # Use cross-validation score (neg_mean_squared_error)
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score  # Minimize MSE (convert to positive MSE)


# Hyperparameter bounds for SGB
lb_sgb = np.array([50, 0.01, 3])
ub_sgb = np.array([200, 0.3, 10])

# Hyperparameter bounds for LGBR
lb_lgbr = np.array([10, 0.01, 50])
ub_lgbr = np.array([100, 0.3, 200])

# Optimize SGB using AOA
print("Optimizing SGB with AOA...")
best_params_sgb, best_score_sgb = aoa_optimize(
    objective_sgb, lb_sgb, ub_sgb, N=20, T=50
)
print(
    f"Best SGB params: n_estimators={int(best_params_sgb[0])}, learning_rate={best_params_sgb[1]:.4f}, max_depth={int(best_params_sgb[2])}"
)
print(f"Best CV MSE: {best_score_sgb:.4f}")

# Train final SGB model and evaluate on test set
sgb_final = GradientBoostingRegressor(
    n_estimators=int(best_params_sgb[0]),
    learning_rate=best_params_sgb[1],
    max_depth=int(best_params_sgb[2]),
    random_state=42,
)
sgb_final.fit(X_train, y_train)
y_pred_sgb = sgb_final.predict(X_test)
test_mse_sgb = mean_squared_error(y_test, y_pred_sgb)
print(f"Test MSE for SGB: {test_mse_sgb:.4f}\n")

# Optimize LGBR using AOA
print("Optimizing LGBR with AOA...")
best_params_lgbr, best_score_lgbr = aoa_optimize(
    objective_lgbr, lb_lgbr, ub_lgbr, N=20, T=50
)
print(
    f"Best LGBR params: num_leaves={int(best_params_lgbr[0])}, learning_rate={best_params_lgbr[1]:.4f}, n_estimators={int(best_params_lgbr[2])}"
)
print(f"Best CV MSE: {best_score_lgbr:.4f}")

# Train final LGBR model and evaluate on test set
lgbr_final = lgb.LGBMRegressor(
    num_leaves=int(best_params_lgbr[0]),
    learning_rate=best_params_lgbr[1],
    n_estimators=int(best_params_lgbr[2]),
    random_state=42,
    verbosity=-1,
)
lgbr_final.fit(X_train, y_train)
y_pred_lgbr = lgbr_final.predict(X_test)
test_mse_lgbr = mean_squared_error(y_test, y_pred_lgbr)
print(f"Test MSE for LGBR: {test_mse_lgbr:.4f}")
