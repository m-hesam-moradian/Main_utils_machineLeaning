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


# Lévy flight function
def levy_flight(D):
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(D) * sigma
    v = np.random.randn(D)
    step = u / np.abs(v) ** (1 / beta)
    return step


# Arctic Puffin Optimization (APO) implementation
def apo_optimize(objective_func, lb, ub, N=30, T=100, F=0.5, C=0.5):
    """
    Arctic Puffin Optimization Algorithm for minimizing an objective function.
    Based on the paper: aerial flight (exploration) and underwater foraging (exploitation) phases.
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
        # Behavioral conversion factor B
        B = 2 * np.log(1 / np.random.rand()) * (1 - t / T)

        for i in range(N):
            if B > C:  # Aerial flight (Exploration)
                # Strategy 1: Aerial search
                r = np.random.randint(0, N)
                while r == i:
                    r = np.random.randint(0, N)
                alpha = 2 * (1 - t / T)
                L = levy_flight(D)
                R = np.round(0.5 * (0.05 + np.random.rand())) * alpha
                Y = population[i] + (population[i] - population[r]) * L + R

                # Strategy 2: Swoop predation
                S = np.tan((np.random.rand() - 0.5) * np.pi)
                Z = Y * S

                # Merge Y and Z, but since single i, evaluate both and choose better
                fit_Y = objective_func(Y)
                fit_Z = objective_func(Z)
                if fit_Y < fit_Z:
                    new_pos = Y
                    new_fit = fit_Y
                else:
                    new_pos = Z
                    new_fit = fit_Z

            else:  # Underwater foraging (Exploitation)
                # Strategy 1: Gathering foraging
                r1, r2, r3 = np.random.choice(N, 3, replace=False)
                if np.random.rand() >= 0.5:
                    W = population[r1] + F * levy_flight(D) * (population[r2] - population[r3])
                else:
                    W = population[r1] + F * (population[r2] - population[r3])

                # Strategy 2: Intensifying search
                f = 0.1 * (np.random.rand() - 1) * (T - t) / T
                Y = W * (1 + f)

                # Strategy 3: Avoiding predators
                r1, r2 = np.random.choice(N, 2, replace=False)
                beta = np.random.rand()
                if np.random.rand() >= 0.5:
                    Z = population[i] + F * levy_flight(D) * (population[r1] - population[r2])
                else:
                    Z = population[i] + beta * (population[r1] - population[r2])

                # Merge W, Y, Z: evaluate all and choose best
                fit_W = objective_func(W)
                fit_Y = objective_func(Y)
                fit_Z = objective_func(Z)
                min_fit = min(fit_W, fit_Y, fit_Z)
                if min_fit == fit_W:
                    new_pos = W
                elif min_fit == fit_Y:
                    new_pos = Y
                else:
                    new_pos = Z
                new_fit = min_fit

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Update if better
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t}: Best params = {best_params}, Best score = {best_score:.4f}")

        # Sort and select top N (but since no population reduction, skip; assume fixed N)

    return best_params, best_score


# Objective function for XGBRegressor (XGBR)
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
        max_depth=md, learning_rate=lr, n_estimators=ne, subsample=ss, random_state=42, verbosity=0
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score  # Minimize MSE


# Objective function for K-Nearest Neighbours (KNN)
def objective_knn(params):
    """
    Objective for KNeighborsRegressor.
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


# Hyperparameter bounds for XGBR
lb_xgbr = np.array([3, 0.01, 50, 0.5])
ub_xgbr = np.array([10, 0.3, 200, 1.0])

# Hyperparameter bounds for KNN
lb_knn = np.array([3, 0, 1])
ub_knn = np.array([30, 1, 2])

# Hyperparameter bounds for DT
lb_dt = np.array([3, 2, 1])
ub_dt = np.array([20, 20, 10])

# Optimize XGBR using APO
print("Optimizing XGBR with APO...")
best_params_xgbr, best_score_xgbr = apo_optimize(
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
    random_state=42,
    verbosity=0
)
xgbr_final.fit(X_train, y_train)
y_pred_xgbr = xgbr_final.predict(X_test)
test_mse_xgbr = mean_squared_error(y_test, y_pred_xgbr)
print(f"Test MSE for XGBR: {test_mse_xgbr:.4f}\n")

# Optimize KNN using APO
print("Optimizing KNN with APO...")
best_params_knn, best_score_knn = apo_optimize(
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

# Optimize DT using APO
print("Optimizing DT with APO...")
best_params_dt, best_score_dt = apo_optimize(
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