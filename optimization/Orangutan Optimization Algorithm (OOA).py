import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# =====================================================
# 1. Load your local Excel dataset
# =====================================================
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

print(f"Loading dataset from: {file_path}")
df = pd.read_excel(file_path)

# Assume last column = target (continuous), others = features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Dataset loaded → Features shape: {X.shape}, Target shape: {y.shape}")

# Scale features (highly recommended)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================================
# 2. Catch Fish Optimization Algorithm (CFOA) - based on Arctic Puffin
# =====================================================
def cfoa_optimize(objective_func, lb, ub, N=25, T=70):
    """
    Catch Fish Optimization Algorithm (CFOA)
    Inspired by Arctic Puffin behaviors (from paper):
    - Aerial flight (exploration): Levy flight + swoop predation
    - Underwater foraging (exploitation): gathering, intensifying, predator avoidance
    """
    start_time = time.time()
    D = len(lb)
    # Initialize population
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]
    print(f"Iteration 0: Best params = {best_params.round(4)}, Best MSE = {best_score:.4f}")

    convergence = [best_score]  # For MBE convergence plot

    for t in range(1, T + 1):
        B = 2 * np.log(1 / np.random.rand()) * (1 - t / T)  # Behavioral conversion factor

        for i in range(N):
            if B > 0.5:  # Aerial flight phase (Exploration)
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

                # Choose better
                fit_Y = objective_func(Y)
                fit_Z = objective_func(Z)
                new_pos = Y if fit_Y < fit_Z else Z
                new_fit = min(fit_Y, fit_Z)
            else:  # Underwater foraging phase (Exploitation)
                r1, r2, r3 = np.random.choice(range(N), 3, replace=False)
                # Gathering foraging
                if np.random.rand() >= 0.5:
                    W = population[r1] + 0.5 * levy_flight(D) * (population[r2] - population[r3])
                else:
                    W = population[r1] + 0.5 * (population[r2] - population[r3])

                # Intensifying search
                f = 0.1 * (np.random.rand() - 1) * (T - t) / T
                Y = W * (1 + f)

                # Predator avoidance
                r4, r5 = np.random.choice(range(N), 2, replace=False)
                beta = np.random.rand()
                if np.random.rand() >= 0.5:
                    Z = population[i] + 0.5 * levy_flight(D) * (population[r4] - population[r5])
                else:
                    Z = population[i] + beta * (population[r4] - population[r5])

                # Choose best among W, Y, Z
                fit_W = objective_func(W)
                fit_Y = objective_func(Y)
                fit_Z = objective_func(Z)
                min_fit = min(fit_W, fit_Y, fit_Z)
                new_pos = W if min_fit == fit_W else (Y if min_fit == fit_Y else Z)
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
                    print(f"Iteration {t:2d} → Best params = {best_params.round(4)}, MSE = {best_score:.4f}")

            convergence.append(best_score)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nTotal runtime: {runtime:.2f} seconds\n")

    # Convergence plot (MBE = Mean Best Error over iterations)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(convergence)), convergence, label='Best MSE per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Best MSE (MBE)')
    plt.title('Convergence Curve - Catch Fish Optimization Algorithm')
    plt.legend()
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
    alpha = params[0]
    model = Ridge(alpha=alpha)
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Objective function for Extra Trees Regression (ETR)
def objective_etr(params):
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


# Hyperparameter bounds
lb_rr = np.array([0.01])
ub_rr = np.array([100])

lb_etr = np.array([50, 3, 2, 1])
ub_etr = np.array([300, 30, 20, 10])

lb_hgbr = np.array([50, 0.01, 3, 1])
ub_hgbr = np.array([300, 0.3, 15, 20])


# =====================================================
# Run CFOA on all models
# =====================================================

print("\n" + "="*90)
print("CATCH FISH OPTIMIZATION ALGORITHM (CFOA) - 2025")
print("="*90)

# 1. Ridge Regression
print("\nOptimizing Ridge Regression (RR)...")
best_rr, score_rr = cfoa_optimize(objective_rr, lb_rr, ub_rr, N=25, T=60)

print("\nBest RR parameters found:")
print(f"  alpha = {best_rr[0]:.4f}")
print(f"  Best CV MSE = {score_rr:.4f}")

rr_final = Ridge(alpha=best_rr[0])
rr_final.fit(X_train, y_train)
test_mse_rr = mean_squared_error(y_test, rr_final.predict(X_test))
print(f"Final Test MSE (RR) = {test_mse_rr:.4f}\n")


# 2. Extra Trees Regression
print("\nOptimizing Extra Trees Regression (ETR)...")
best_etr, score_etr = cfoa_optimize(objective_etr, lb_etr, ub_etr, N=25, T=60)

print("\nBest ETR parameters found:")
print(f"  n_estimators     = {int(best_etr[0])}")
print(f"  max_depth        = {int(best_etr[1])}")
print(f"  min_samples_split = {int(best_etr[2])}")
print(f"  min_samples_leaf  = {int(best_etr[3])}")
print(f"  Best CV MSE       = {score_etr:.4f}")

etr_final = ExtraTreesRegressor(
    n_estimators=int(best_etr[0]),
    max_depth=int(best_etr[1]),
    min_samples_split=int(best_etr[2]),
    min_samples_leaf=int(best_etr[3]),
    random_state=42
)
etr_final.fit(X_train, y_train)
test_mse_etr = mean_squared_error(y_test, etr_final.predict(X_test))
print(f"Final Test MSE (ETR) = {test_mse_etr:.4f}\n")


# 3. Histogram-Based Gradient Boosting Regression
print("\nOptimizing Histogram-Based Gradient Boosting Regression (HGBR)...")
best_hgbr, score_hgbr = cfoa_optimize(objective_hgbr, lb_hgbr, ub_hgbr, N=25, T=80)

print("\nBest HGBR parameters found:")
print(f"  max_iter         = {int(best_hgbr[0])}")
print(f"  learning_rate    = {best_hgbr[1]:.4f}")
print(f"  max_depth        = {int(best_hgbr[2])}")
print(f"  min_samples_leaf = {int(best_hgbr[3])}")
print(f"  Best CV MSE      = {score_hgbr:.4f}")

hgbr_final = HistGradientBoostingRegressor(
    max_iter=int(best_hgbr[0]),
    learning_rate=best_hgbr[1],
    max_depth=int(best_hgbr[2]),
    min_samples_leaf=int(best_hgbr[3]),
    random_state=42
)
hgbr_final.fit(X_train, y_train)
test_mse_hgbr = mean_squared_error(y_test, hgbr_final.predict(X_test))
print(f"Final Test MSE (HGBR) = {test_mse_hgbr:.4f}\n")