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


# Heavy Rain Optimization Algorithm (HROA) implementation
def hroa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Heavy Rain Optimization Algorithm for minimizing an objective function.
    Inspired by heavy rain phenomena: cloud burst (exploration), rainfall (movement to minima), flooding (exploitation), and evaporation (diversification).
    """
    D = len(lb)
    # Initialize population (raindrops)
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        # Adaptive rain intensity: exploration to exploitation
        rain_intensity = 1 - t / T

        for i in range(N):
            # Phase 1: Cloud Burst (Exploration: large random drops)
            if np.random.rand() < rain_intensity:
                drop_size = (ub - lb) * np.random.uniform(0.1, 0.5, D)
                direction = np.sign(np.random.randn(D))
                new_pos = population[i] + direction * np.random.rand(D) * drop_size
            else:
                # Phase 2: Rainfall (Exploitation: flow to minima)
                r = np.random.rand(D)
                flow_factor = (1 - rain_intensity) * r
                new_pos = population[i] + flow_factor * (best_params - population[i])

            # Phase 3: Flooding (Local intensification)
            flood = (ub - lb) * np.random.randn(D) * (1 / (t + 1))
            new_pos += flood

            # Phase 4: Evaporation (Diversification: vapor rise if flooded)
            if np.random.rand() < 0.1 * rain_intensity:
                vapor_strength = (ub - lb) * np.random.uniform(0.05, 0.2, D)
                new_pos += vapor_strength * np.random.randn(D)

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

        # Phase 5: Storm Clearance (Elite replacement to refresh)
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

# Optimize SVR using HROA
print("Optimizing SVR with HROA...")
best_params_svr, best_score_svr = hroa_optimize(
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

# Optimize XGBR using HROA
print("Optimizing XGBR with HROA...")
best_params_xgbr, best_score_xgbr = hroa_optimize(
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