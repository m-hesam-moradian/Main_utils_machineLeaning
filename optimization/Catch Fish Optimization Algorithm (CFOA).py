import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset from local Excel file
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
print(f"Loading dataset from {file_path}...")
df = pd.read_excel(file_path)

# Assume last column is target, others are features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Catch Fish Optimization Algorithm (CFOA) implementation
def cfoa_optimize(objective_func, lb, ub, N=30, T=100, F=0.5, C=0.5):
    """
    Catch Fish Optimization Algorithm (CFOA) for minimizing an objective function.
    Based on Arctic Puffin behaviors from the PDF: aerial flight (exploration) and underwater foraging (exploitation).
    """
    start_time = time.time()
    D = len(lb)
    population = lb + np.random.rand(N, D) * (ub - lb)
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    convergence = [best_score]  # For convergence plot

    for t in range(1, T + 1):
        B = 2 * np.log(1 / np.random.rand()) * (1 - t / T)  # Behavioral conversion factor

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

                # Choose better between Y and Z
                fit_Y = objective_func(Y)
                fit_Z = objective_func(Z)
                if fit_Y < fit_Z:
                    new_pos = Y
                    new_fit = fit_Y
                else:
                    new_pos = Z
                    new_fit = fit_Z
            else:  # Underwater foraging (Exploitation)
                r1, r2, r3 = np.random.choice(range(N), 3, replace=False)
                # Strategy 1: Gathering foraging
                if np.random.rand() >= 0.5:
                    W = population[r1] + F * levy_flight(D) * (population[r2] - population[r3])
                else:
                    W = population[r1] + F * (population[r2] - population[r3])

                # Strategy 2: Intensifying search
                f = 0.1 * (np.random.rand() - 1) * (T - t) / T
                Y = W * (1 + f)

                # Strategy 3: Avoiding predators
                r4, r5 = np.random.choice(range(N), 2, replace=False)
                beta = np.random.rand()
                if np.random.rand() >= 0.5:
                    Z = population[i] + F * levy_flight(D) * (population[r4] - population[r5])
                else:
                    Z = population[i] + beta * (population[r4] - population[r5])

                # Choose the best among W, Y, Z
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

        convergence.append(best_score)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")

    # Convergence plot (MBE = Mean Best Error over iterations)
    plt.figure(figsize=(8, 5))
    plt.plot(range(T+1), convergence)
    plt.xlabel('Iteration')
    plt.ylabel('Best MSE (MBE)')
    plt.title('Convergence Curve (Mean Best Error over Iterations)')
    plt.grid(True)
    plt.show()

    return best_params, best_score


# Levy flight function
def levy_flight(D):
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(D) * sigma
    v = np.random.randn(D)
    step = u / np.abs(v) ** (1 / beta)
    return step


# Objective function for Ridge Regression (RR)
def objective_rr(params):
    """
    Objective for Ridge.
    Params: [alpha]
    """
    alpha = params[0]

    model = Ridge(
        alpha=alpha
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Objective function for Extra Trees Regression (ETR)
def objective_etr(params):
    """
    Objective for ExtraTreesRegressor.
    Params: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
    """
    ne = int(params[0])
    md = int(params[1])
    mss = int(params[2])
    msl = int(params[3])

    model = ExtraTreesRegressor(
        n_estimators=ne, max_depth=md, min_samples_split=mss, min_samples_leaf=msl, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


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
    return -score


# Hyperparameter bounds for RR
lb_rr = np.array([0.01])
ub_rr = np.array([100])

# Hyperparameter bounds for ETR
lb_etr = np.array([50, 3, 2, 1])
ub_etr = np.array([300, 30, 20, 10])

# Hyperparameter bounds for HGBR
lb_hgbr = np.array([50, 0.01, 3, 1])
ub_hgbr = np.array([300, 0.3, 15, 20])


# Optimize RR using CFOA
print("Optimizing RR with CFOA...")
best_params_rr, best_score_rr = cfoa_optimize(
    objective_rr, lb_rr, ub_rr, N=20, T=50
)
print(
    f"Best RR params: alpha={best_params_rr[0]:.4f}"
)
print(f"Best CV MSE: {best_score_rr:.4f}")

# Train final RR model and evaluate on test set
rr_final = Ridge(alpha=best_params_rr[0])
rr_final.fit(X_train, y_train)
y_pred_rr = rr_final.predict(X_test)
test_mse_rr = mean_squared_error(y_test, y_pred_rr)
print(f"Test MSE for RR: {test_mse_rr:.4f}\n")

# Optimize ETR using CFOA
print("Optimizing ETR with CFOA...")
best_params_etr, best_score_etr = cfoa_optimize(
    objective_etr, lb_etr, ub_etr, N=20, T=50
)
print(
    f"Best ETR params: n_estimators={int(best_params_etr[0])}, max_depth={int(best_params_etr[1])}, "
    f"min_samples_split={int(best_params_etr[2])}, min_samples_leaf={int(best_params_etr[3])}"
)
print(f"Best CV MSE: {best_score_etr:.4f}")

# Train final ETR model and evaluate on test set
etr_final = ExtraTreesRegressor(
    n_estimators=int(best_params_etr[0]),
    max_depth=int(best_params_etr[1]),
    min_samples_split=int(best_params_etr[2]),
    min_samples_leaf=int(best_params_etr[3]),
    random_state=42
)
etr_final.fit(X_train, y_train)
y_pred_etr = etr_final.predict(X_test)
test_mse_etr = mean_squared_error(y_test, y_pred_etr)
print(f"Test MSE for ETR: {test_mse_etr:.4f}\n")

# Optimize HGBR using CFOA
print("Optimizing HGBR with CFOA...")
best_params_hgbr, best_score_hgbr = cfoa_optimize(
    objective_hgbr, lb_hgbr, ub_hgbr, N=20, T=50
)
print(
    f"Best HGBR params: max_iter={int(best_params_hgbr[0])}, learning_rate={best_params_hgbr[1]:.4f}, "
    f"max_depth={int(best_params_hgbr[2])}, min_samples_leaf={int(best_params_hgbr[3])}"
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
print(f"Test MSE for HGBR: {test_mse_hgbr:.4f}")