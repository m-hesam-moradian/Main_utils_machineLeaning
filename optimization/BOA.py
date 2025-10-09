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


# Botox Optimization Algorithm (BOA) implementation
def boa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Botox Optimization Algorithm for minimizing an objective function.

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
    X = np.zeros((N, D))
    for j in range(D):
        X[:, j] = lb[j] + np.random.rand(N) * (ub[j] - lb[j])

    # Evaluate initial population
    F = np.array([objective_func(x) for x in X])
    best_idx = np.argmin(F)
    X_best = X[best_idx].copy()
    f_best = F[best_idx]

    for t in range(1, T + 1):
        # Number of variables to update
        M = int(D * (1 - t / T))
        if M < 1:
            M = 1  # Ensure at least one variable is updated

        # Mean population position
        X_mean = np.mean(X, axis=0)

        # Botox amount (same for all individuals)
        B = (X_mean - X_best) * (t / T)

        for i in range(N):
            # Select M unique dimensions randomly
            S_i = np.random.choice(D, M, replace=False)

            # Create new candidate
            X_new = X[i].copy()
            for jk in S_i:
                X_new[jk] = X[i][jk] + np.random.rand() * B[jk]

            # Clip to bounds
            X_new = np.clip(X_new, lb, ub)

            # Evaluate
            f_new = objective_func(X_new)

            # Update if better
            if f_new < F[i]:
                X[i] = X_new
                F[i] = f_new
                if f_new < f_best:
                    X_best = X_new.copy()
                    f_best = f_new

    return X_best, f_best


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

# Optimize SGB using BOA
print("Optimizing SGB with BOA...")
best_params_sgb, best_score_sgb = boa_optimize(
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

# Optimize LGBR using BOA
print("Optimizing LGBR with BOA...")
best_params_lgbr, best_score_lgbr = boa_optimize(
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
