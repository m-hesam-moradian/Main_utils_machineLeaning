import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Load sample dataset for demonstration
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Football Optimization Algorithm (FOA) implementation
def foa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Football Optimization Algorithm for minimizing an objective function.
    Inspired by football (soccer) team strategies: scouting (exploration), passing (convergence to best), shooting (local refinement), defense (avoid bad positions), and substitution (diversification).
    """
    D = len(lb)
    # Initialize population (players on the field)
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        # Adaptive play intensity: exploration to exploitation
        play_intensity = 1 - t / T

        for i in range(N):
            # Phase 1: Scouting (Exploration: search for new talent/positions)
            if np.random.rand() < play_intensity:
                scout_step = (ub - lb) * np.random.uniform(0.1, 0.4, D)
                direction = np.sign(np.random.randn(D))
                new_pos = population[i] + direction * np.random.rand(D) * scout_step
            else:
                # Phase 2: Passing (Exploitation: pass to star player/best)
                r = np.random.rand(D)
                pass_factor = (1 - play_intensity) * r
                new_pos = population[i] + pass_factor * (best_params - population[i])

            # Phase 3: Shooting (Local refinement for goal)
            shot = (ub - lb) * np.random.randn(D) * (1 / (t + 1))
            new_pos += shot

            # Phase 4: Defense (Avoid bad positions/diversify)
            if np.random.rand() < 0.1 * play_intensity:
                defense_strength = (ub - lb) * np.random.uniform(0.05, 0.15, D)
                new_pos += defense_strength * np.random.randn(D)

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

        # Phase 5: Substitution (Replace underperformer)
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.05:
            population[worst_idx] = lb + np.random.rand(D) * (ub - lb)
            fitness[worst_idx] = objective_func(population[worst_idx])

    return best_params, best_score


# Objective function for XGBR
def objective_xgbr(params):
    """
    Objective for XGBRegressor (XGBR).
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
    return -score  # Minimize MSE


# Objective function for K-Nearest Neighbours (KNN)
def objective_knn(params):
    """
    Objective for KNeighborsRegressor (KNN).
    Params: [n_neighbors, weights_index (0: uniform, 1: distance), p]
    """
    nn = int(params[0])
    wi = int(params[1])
    p = int(params[2])

    weights = ['uniform', 'distance'][wi]

    model = KNeighborsRegressor(
        n_neighbors=nn, weights=weights, p=p
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Objective function for Decision Trees (DT)
def objective_dt(params):
    """
    Objective for DecisionTreeRegressor (DT).
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


# Hyperparameter bounds for XGBR
lb_xgbr = np.array([3, 0.01, 50, 0.5])
ub_xgbr = np.array([15, 0.3, 300, 1.0])

# Hyperparameter bounds for KNN
lb_knn = np.array([3, 0, 1])
ub_knn = np.array([30, 1, 2])

# Hyperparameter bounds for DT
lb_dt = np.array([3, 2, 1])
ub_dt = np.array([20, 20, 10])

# Optimize XGBR using FOA
print("Optimizing XGBR with FOA...")
best_params_xgbr, best_score_xgbr = foa_optimize(
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
print(f"Test MSE for XGBR: {test_mse_xgbr:.4f}\n")

# Optimize KNN using FOA
print("Optimizing KNN with FOA...")
best_params_knn, best_score_knn = foa_optimize(
    objective_knn, lb_knn, ub_knn, N=20, T=50
)
print(
    f"Best KNN params: n_neighbors={int(best_params_knn[0])}, weights={['uniform', 'distance'][int(best_params_knn[1])]}, p={int(best_params_knn[2])}"
)
print(f"Best CV MSE: {best_score_knn:.4f}")

# Train final KNN model and evaluate on test set
weights = ['uniform', 'distance'][int(best_params_knn[1])]
knn_final = KNeighborsRegressor(
    n_neighbors=int(best_params_knn[0]),
    weights=weights,
    p=int(best_params_knn[2])
)
knn_final.fit(X_train, y_train)
y_pred_knn = knn_final.predict(X_test)
test_mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"Test MSE for KNN: {test_mse_knn:.4f}\n")

# Optimize DT using FOA
print("Optimizing DT with FOA...")
best_params_dt, best_score_dt = foa_optimize(
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