import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pandas as pd
# Load sample dataset for demonstration
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.22-Data.xlsx"
sheet_name = "Z-score"
target_column = "Renewable Availability Index"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Spider Wasp Optimizer (SWO) implementation
def swo_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Spider Wasp Optimizer for minimizing an objective function.
    Inspired by spider wasp hunting behavior: exploration (searching for prey),
    paralysis (targeting best solution), and dragging (exploitation to nest).

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
    # Initialize population (wasps)
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        a = 2 - t * (2 / T)  # Linearly decreasing from 2 to 0
        
        # Sort population by fitness
        sorted_idx = np.argsort(fitness)
        
        for i in range(N):
            # Phase 1: Searching (Exploration - random walk with large steps)
            if np.random.rand() < 0.5:
                r1 = np.random.rand(D)
                r2 = np.random.rand(D)
                A = 2 * a * r1
                C = 2 * r2
                # Move away from random position for exploration
                rand_idx = np.random.randint(0, N)
                new_pos = population[i] - A * np.abs(C * population[rand_idx] - population[i])
            else:
                # Levy flight exploration
                levy_step = 0.01 * np.random.normal(0, 1, D) * (ub - lb)
                new_pos = population[i] + levy_step * (np.random.rand(D) - 0.5)

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

            # Phase 2: Paralysis Attack (Exploitation - target best solution)
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            A = 2 * a * r1
            C = 2 * r2
            new_pos = best_params - A * np.abs(C * best_params - population[i])
            
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

            # Phase 3: Dragging to Nest (Intensification - elite opposition)
            if np.random.rand() < 0.1:  # Occasional elite strategy
                elite_opp = lb + ub - best_params  # Elite opposition
                r = np.random.rand(D)
                new_pos = population[i] + r * (elite_opp - population[i])
                
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

# Optimize SGB using SWO
print("Optimizing SGB with SWO...")
best_params_sgb, best_score_sgb = swo_optimize(
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

# Optimize LGBR using SWO
print("Optimizing LGBR with SWO...")
best_params_lgbr, best_score_lgbr = swo_optimize(
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