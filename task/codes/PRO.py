import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load sample dataset for demonstration
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Partial Reinforcement Optimizer (PRO) implementation
def pro_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Partial Reinforcement Optimizer for minimizing an objective function.

    Inspired by the partial reinforcement effect (PRE) in psychology, where intermittent
    reinforcement leads to more persistent behaviors resistant to extinction.

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

    # Reinforcement counters for each individual (tracks recent successes)
    reinforcement = np.zeros(N)

    for t in range(1, T + 1):
        # Decreasing exploration factor
        alpha = 1 - (t / T)

        for i in range(N):
            # Determine behavior based on reinforcement level
            if reinforcement[i] > 0:
                # Persistent exploitation (continue towards best due to prior reinforcement)
                r = np.random.rand(D)
                step_size = alpha * np.random.uniform(0.5, 1.0)
                new_pos = population[i] + step_size * r * (best_params - population[i])

                # Clip to bounds
                new_pos = np.clip(new_pos, lb, ub)

                # Evaluate
                new_fit = objective_func(new_pos)

                # Partial reinforcement: accept if better, or sometimes even if not (to persist)
                accept_prob = 0.2 + 0.3 * (
                    reinforcement[i] / 5
                )  # Higher reinforcement, higher accept prob
                if new_fit < fitness[i] or np.random.rand() < accept_prob:
                    population[i] = new_pos
                    fitness[i] = new_fit
                    reinforcement[i] = min(
                        reinforcement[i] + 1, 5
                    )  # Strengthen reinforcement (cap at 5)
                    if new_fit < best_score:
                        best_params = new_pos.copy()
                        best_score = new_fit
                else:
                    reinforcement[i] -= 1  # Weaken if no improvement

            else:
                # Exploration phase (acquisition of new behaviors)
                r = np.random.rand(D)
                new_pos = (
                    population[i] + alpha * (2 * r - 1) * (ub - lb) / 10
                )  # Small random perturbation

                # Or with low prob, large jump
                if np.random.rand() < 0.1:
                    new_pos = lb + np.random.rand(D) * (ub - lb)

                # Clip to bounds
                new_pos = np.clip(new_pos, lb, ub)

                # Evaluate
                new_fit = objective_func(new_pos)

                # Reinforce intermittently if better
                if new_fit < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fit
                    # Variable ratio reinforcement: start with random count
                    reinforcement[i] = np.random.randint(1, 4)
                    if new_fit < best_score:
                        best_params = new_pos.copy()
                        best_score = new_fit
                else:
                    # Occasional reinforcement to prevent early extinction
                    if np.random.rand() < 0.05:
                        reinforcement[i] = 1

            # Ensure reinforcement doesn't go negative
            reinforcement[i] = max(reinforcement[i], 0)

        # Global best update
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_score:
            best_params = population[best_idx].copy()
            best_score = fitness[best_idx]

    return best_params, best_score


# Objective function for Decision Tree Regression (DTR)
def objective_dtr(params):
    """Objective for DecisionTreeRegressor.
    Params: [max_depth, min_samples_split, min_samples_leaf]
    """
    max_depth = int(params[0]) if params[0] > 0 else None
    min_samples_split = int(params[1])
    min_samples_leaf = int(params[2])

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score  # Minimize MSE


# Objective function for LightGBM Regressor (LGBR)
def objective_lgbr(params):
    """Objective for LGBMRegressor.
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
        verbosity=-1,
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Objective function for AdaBoost Regression (ADAR)
def objective_adar(params):
    """Objective for AdaBoostRegressor.
    Params: [n_estimators, learning_rate, base_estimator_depth]
    """
    n_est = int(params[0])
    lr = params[1]
    base_depth = int(params[2]) if params[2] > 0 else None

    # Create base estimator (Decision Tree)
    base_estimator = DecisionTreeRegressor(max_depth=base_depth, random_state=42)

    model = AdaBoostRegressor(
        estimator=base_estimator, n_estimators=n_est, learning_rate=lr, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Hyperparameter bounds
# DTR bounds
lb_dtr = np.array([3, 2, 1])
ub_dtr = np.array([15, 20, 10])

# LGBR bounds
lb_lgbr = np.array([10, 0.01, 50])
ub_lgbr = np.array([100, 0.3, 200])

# ADAR bounds
lb_adar = np.array([50, 0.01, 3])
ub_adar = np.array([200, 1.0, 10])

# Optimize DTR using PRO
print("Optimizing Decision Tree Regression (DTR) with PRO...")
best_params_dtr, best_score_dtr = pro_optimize(
    objective_dtr, lb_dtr, ub_dtr, N=20, T=50
)
print(
    f"Best DTR params: max_depth={int(best_params_dtr[0]) if best_params_dtr[0] > 0 else 'None'}, "
    f"min_samples_split={int(best_params_dtr[1])}, min_samples_leaf={int(best_params_dtr[2])}"
)
print(f"Best CV MSE for DTR: {best_score_dtr:.4f}")

# Train final DTR model
dtr_final = DecisionTreeRegressor(
    max_depth=int(best_params_dtr[0]) if best_params_dtr[0] > 0 else None,
    min_samples_split=int(best_params_dtr[1]),
    min_samples_leaf=int(best_params_dtr[2]),
    random_state=42,
)
dtr_final.fit(X_train, y_train)
y_pred_dtr = dtr_final.predict(X_test)
test_mse_dtr = mean_squared_error(y_test, y_pred_dtr)
print(f"Test MSE for DTR: {test_mse_dtr:.4f}\n")

# Optimize LGBR using PRO
print("Optimizing LightGBM Regression (LGBR) with PRO...")
best_params_lgbr, best_score_lgbr = pro_optimize(
    objective_lgbr, lb_lgbr, ub_lgbr, N=20, T=50
)
print(
    f"Best LGBR params: num_leaves={int(best_params_lgbr[0])}, "
    f"learning_rate={best_params_lgbr[1]:.4f}, n_estimators={int(best_params_lgbr[2])}"
)
print(f"Best CV MSE for LGBR: {best_score_lgbr:.4f}")

# Train final LGBR model
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
print(f"Test MSE for LGBR: {test_mse_lgbr:.4f}\n")

# Optimize ADAR using PRO
print("Optimizing AdaBoost Regression (ADAR) with PRO...")
best_params_adar, best_score_adar = pro_optimize(
    objective_adar, lb_adar, ub_adar, N=20, T=50
)
print(
    f"Best ADAR params: n_estimators={int(best_params_adar[0])}, "
    f"learning_rate={best_params_adar[1]:.4f}, base_depth={int(best_params_adar[2]) if best_params_adar[2] > 0 else 'None'}"
)
print(f"Best CV MSE for ADAR: {best_score_adar:.4f}")

# Train final ADAR model
base_est = DecisionTreeRegressor(
    max_depth=int(best_params_adar[2]) if best_params_adar[2] > 0 else None,
    random_state=42,
)
adar_final = AdaBoostRegressor(
    estimator=base_est,
    n_estimators=int(best_params_adar[0]),
    learning_rate=best_params_adar[1],
    random_state=42,
)
adar_final.fit(X_train, y_train)
y_pred_adar = adar_final.predict(X_test)
test_mse_adar = mean_squared_error(y_test, y_pred_adar)
print(f"Test MSE for ADAR: {test_mse_adar:.4f}")

# Summary comparison
print("\n" + "=" * 60)
print("PRO OPTIMIZATION SUMMARY")
print("=" * 60)
models_summary = {"DTR": test_mse_dtr, "LGBR": test_mse_lgbr, "ADAR": test_mse_adar}
best_model = min(models_summary, key=models_summary.get)
print(
    f"Best performing model: {best_model} (Test MSE: {models_summary[best_model]:.4f})"
)
for model, mse in models_summary.items():
    print(f"{model:4s}: Test MSE = {mse:.4f}")
