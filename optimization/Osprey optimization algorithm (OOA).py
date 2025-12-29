import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
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


# Osprey Optimization Algorithm (OOA) implementation
def ooa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Osprey Optimization Algorithm for minimizing an objective function.
    Based on osprey hunting behavior: exploration (detecting and hunting fish) and exploitation (carrying fish to suitable position).
    """
    start_time = time.time()  # Start timing
    D = len(lb)
    # Initialize population
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]
    print(f"Iteration 0: Best params = {best_params}, Best score = {best_score:.4f}")

    for t in range(1, T + 1):
        for i in range(N):
            # Phase 1: Exploration (Position identification and hunting)
            # Find better solutions as "fish"
            better_indices = np.where(fitness < fitness[i])[0]
            if len(better_indices) > 0:
                fish_idx = np.random.choice(better_indices)
                SF = population[fish_idx]
            else:
                SF = best_params  # Use best if no better

            r = np.random.rand(D)
            I = np.random.choice([1, 2], size=D)
            new_pos_p1 = population[i] + r * (SF - I * population[i])

            # Clip to bounds
            new_pos_p1 = np.clip(new_pos_p1, lb, ub)

            # Evaluate and update if better
            new_fit_p1 = objective_func(new_pos_p1)
            if new_fit_p1 < fitness[i]:
                population[i] = new_pos_p1
                fitness[i] = new_fit_p1
                if new_fit_p1 < best_score:
                    best_params = new_pos_p1.copy()
                    best_score = new_fit_p1
                    print(f"Iteration {t}: Best params = {best_params}, Best score = {best_score:.4f}")

            # Phase 2: Exploitation (Carrying the fish to suitable position)
            r = np.random.rand(D)
            new_pos_p2 = population[i] + (lb + r * (ub - lb)) / t

            # Clip to bounds
            new_pos_p2 = np.clip(new_pos_p2, lb, ub)

            # Evaluate and update if better
            new_fit_p2 = objective_func(new_pos_p2)
            if new_fit_p2 < fitness[i]:
                population[i] = new_pos_p2
                fitness[i] = new_fit_p2
                if new_fit_p2 < best_score:
                    best_params = new_pos_p2.copy()
                    best_score = new_fit_p2
                    print(f"Iteration {t}: Best params = {best_params}, Best score = {best_score:.4f}")

    end_time = time.time()  # End timing
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")

    return best_params, best_score


# Objective function for Histogram-Based Gradient Boosting Regression (HGBR)
def objective_hgbr(params):
    """
    Objective for HistGradientBoostingRegressor.
    Params: [max_iter, learning_rate, max_depth, min_samples_leaf]
    """
    mi = int(params[0])
    lr = params[1]
    md = int(params[2])
    msl = int(params[3])

    model = HistGradientBoostingRegressor(
        max_iter=mi, learning_rate=lr, max_depth=md, min_samples_leaf=msl, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score  # Minimize MSE


# Objective function for Decision Tree Regression (DTR)
def objective_dtr(params):
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


# Hyperparameter bounds for HGBR
lb_hgbr = np.array([50, 0.01, 3, 1])
ub_hgbr = np.array([300, 0.3, 15, 20])

# Hyperparameter bounds for DTR
lb_dtr = np.array([3, 2, 1])
ub_dtr = np.array([20, 20, 10])

# Optimize HGBR using OOA
print("Optimizing HGBR with OOA...")
best_params_hgbr, best_score_hgbr = ooa_optimize(
    objective_hgbr, lb_hgbr, ub_hgbr, N=20, T=50
)
print(
    f"Best HGBR params: max_iter={int(best_params_hgbr[0])}, learning_rate={best_params_hgbr[1]:.4f}, max_depth={int(best_params_hgbr[2])}, min_samples_leaf={int(best_params_hgbr[3])}"
)
print(f"Best CV MSE: {best_score_hgbr:.4f}")

# Train final HGBR model and evaluate on test set
hgbr_final = HistGradientBoostingRegressor(
    max_iter=int(best_params_hgbr[0]),
    learning_rate=best_params_hgbr[1],
    max_depth=int(best_params_hgbr[2]),
    min_samples_leaf=int(best_params_hgbr[3]),
    random_state=42
)
hgbr_final.fit(X_train, y_train)
y_pred_hgbr = hgbr_final.predict(X_test)
test_mse_hgbr = mean_squared_error(y_test, y_pred_hgbr)
print(f"Test MSE for HGBR: {test_mse_hgbr:.4f}\n")

# Optimize DTR using OOA
print("Optimizing DTR with OOA...")
best_params_dtr, best_score_dtr = ooa_optimize(
    objective_dtr, lb_dtr, ub_dtr, N=20, T=50
)
print(
    f"Best DTR params: max_depth={int(best_params_dtr[0])}, min_samples_split={int(best_params_dtr[1])}, min_samples_leaf={int(best_params_dtr[2])}"
)
print(f"Best CV MSE: {best_score_dtr:.4f}")

# Train final DTR model and evaluate on test set
dtr_final = DecisionTreeRegressor(
    max_depth=int(best_params_dtr[0]),
    min_samples_split=int(best_params_dtr[1]),
    min_samples_leaf=int(best_params_dtr[2]),
    random_state=42
)
dtr_final.fit(X_train, y_train)
y_pred_dtr = dtr_final.predict(X_test)
test_mse_dtr = mean_squared_error(y_test, y_pred_dtr)
print(f"Test MSE for DTR: {test_mse_dtr:.4f}")