import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLars
from sklearn.ensemble import GradientBoostingRegressor
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


# Greylag Goose Optimization (GGO) implementation
def ggo_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Greylag Goose Optimization (GGO) for minimizing an objective function.
    Based on migratory and foraging behaviors of Greylag geese, with dynamic grouping for exploration and exploitation.
    """
    start_time = time.time()
    D = len(lb)
    population = lb + np.random.rand(N, D) * (ub - lb)
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    # Initial group sizes
    Ge_size = N // 2
    Gx_size = N - Ge_size

    stagnation_counter = 0
    prev_best_score = best_score

    convergence = [best_score]  # For MBE convergence plot

    for t in range(1, T + 1):
        a = 2 * (1 - t / T)

        # Sort indices by fitness (ascending, minimize)
        sorted_idx = np.argsort(fitness)
        Ge_idx = sorted_idx[-Ge_size:]  # Worse fitness for exploration
        Gx_idx = sorted_idx[:Gx_size]   # Better fitness for exploitation

        # Exploration phase
        for i in Ge_idx:
            if np.random.rand() < 0.5:
                # Basic exploration
                r1 = np.random.rand(D)
                r2 = np.random.rand(D)
                A = 2 * a * r1
                C = 2 * r2
                new_pos = best_params - A * np.abs(C * best_params - population[i])
            else:
                # Enhanced exploration with three random agents
                rand_idx = np.random.choice(range(N), 3, replace=False)
                X_r1, X_r2, X_r3 = population[rand_idx]
                z = np.exp(-t / T)
                r = np.random.rand(D)
                weights = np.random.dirichlet(np.ones(3), size=1)[0]
                weighted_sum = weights[0] * X_r1 + weights[1] * X_r2 + weights[2] * X_r3
                new_pos = population[i] + z * (r * weighted_sum - population[i])

            # Secondary exploration update
            r_sec = 2 * np.random.rand(D) - 1  # [-1,1]
            A_sec = np.random.uniform(0, 2, D)
            C_sec = np.random.rand(D)
            new_pos = new_pos + a * r_sec * (best_params - new_pos) + C_sec * np.abs(A_sec * best_params - new_pos)

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t}: Best params = {best_params}, Best score = {best_score:.4f}")

        # Exploitation phase
        for i in Gx_idx:
            if np.random.rand() < 0.5:
                # Sentry-guided update (select top 3 best as sentries)
                top_idx = sorted_idx[:3]
                X_s1, X_s2, X_s3 = population[top_idx]
                D1 = np.abs(X_s1 - population[i])
                D2 = np.abs(X_s2 - population[i])
                D3 = np.abs(X_s3 - population[i])
                r = np.random.rand(D)
                a1 = 2 * a * r
                a2 = 2 * a * r
                a3 = 2 * a * r
                new_pos = ( (X_s1 - a1 * D1) + (X_s2 - a2 * D2) + (X_s3 - a3 * D3) ) / 3
            else:
                # Leader-following
                r1 = np.random.rand(D)
                r2 = np.random.rand(D)
                A = 2 * a * r1
                C = 2 * r2
                new_pos = best_params - A * np.abs(C * best_params - population[i])

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t}: Best params = {best_params}, Best score = {best_score:.4f}")

        # Check for stagnation and adjust groups
        if best_score < prev_best_score:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        if stagnation_counter >= 2:
            Ge_size = min(Ge_size + 1, N - 1)
            Gx_size = N - Ge_size
            stagnation_counter = 0
        prev_best_score = best_score

        convergence.append(best_score)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")

    # Convergence plot based on MBE (Mean Best Error)
    plt.figure(figsize=(10, 5))
    plt.plot(range(T+1), convergence)
    plt.xlabel('Iteration')
    plt.ylabel('Best MSE (MBE)')
    plt.title('Convergence Curve - Greylag Goose Optimization (GGO)')
    plt.grid(True)
    plt.show()

    return best_params, best_score


# =====================================================
# Objective functions
# =====================================================

# Lasso Least Angle Regression (LLAR)
def objective_llar(params):
    alpha = params[0]
    mi = int(params[1])

    model = LassoLars(alpha=alpha, max_iter=mi)

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Stochastic Gradient Boosting Regression (SGBR)
def objective_sgbr(params):
    ne = int(params[0])
    lr = params[1]
    md = int(params[2])
    ss = params[3]

    model = GradientBoostingRegressor(
        n_estimators=ne, learning_rate=lr, max_depth=md, subsample=ss, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Histogram-Based Gradient Boosting Regression (HGBR)
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
lb_llar = np.array([0.0001, 500])
ub_llar = np.array([1.0, 5000])

lb_sgbr = np.array([50, 0.01, 3, 0.5])
ub_sgbr = np.array([300, 0.3, 15, 1.0])

lb_hgbr = np.array([50, 0.01, 3, 1])
ub_hgbr = np.array([300, 0.3, 15, 20])


# =====================================================
# Run GGO on all models
# =====================================================

print("\n" + "="*90)
print("GREYLAG GOOSE OPTIMIZATION (GGO) - 2024")
print("="*90)

# 1. LLAR
print("\nOptimizing Lasso Least Angle Regression (LLAR)...")
best_llar, score_llar = ggo_optimize(objective_llar, lb_llar, ub_llar, N=30, T=80)

print("\nBest LLAR parameters found:")
print(f"  alpha    = {best_llar[0]:.6f}")
print(f"  max_iter = {int(best_llar[1])}")
print(f"  Best CV MSE = {score_llar:.4f}")

llar_final = LassoLars(alpha=best_llar[0], max_iter=int(best_llar[1]))
llar_final.fit(X_train, y_train)
test_mse_llar = mean_squared_error(y_test, llar_final.predict(X_test))
print(f"Final Test MSE (LLAR) = {test_mse_llar:.4f}\n")


# 2. SGBR
print("\nOptimizing Stochastic Gradient Boosting Regression (SGBR)...")
best_sgbr, score_sgbr = ggo_optimize(objective_sgbr, lb_sgbr, ub_sgbr, N=30, T=80)

print("\nBest SGBR parameters found:")
print(f"  n_estimators = {int(best_sgbr[0])}")
print(f"  learning_rate = {best_sgbr[1]:.4f}")
print(f"  max_depth = {int(best_sgbr[2])}")
print(f"  subsample = {best_sgbr[3]:.4f}")
print(f"  Best CV MSE = {score_sgbr:.4f}")

sgbr_final = GradientBoostingRegressor(
    n_estimators=int(best_sgbr[0]),
    learning_rate=best_sgbr[1],
    max_depth=int(best_sgbr[2]),
    subsample=best_sgbr[3],
    random_state=42
)
sgbr_final.fit(X_train, y_train)
test_mse_sgbr = mean_squared_error(y_test, sgbr_final.predict(X_test))
print(f"Final Test MSE (SGBR) = {test_mse_sgbr:.4f}\n")


# 3. HGBR
print("\nOptimizing Histogram-Based Gradient Boosting Regression (HGBR)...")
best_hgbr, score_hgbr = ggo_optimize(objective_hgbr, lb_hgbr, ub_hgbr, N=30, T=80)

print("\nBest HGBR parameters found:")
print(f"  max_iter = {int(best_hgbr[0])}")
print(f"  learning_rate = {best_hgbr[1]:.4f}")
print(f"  max_depth = {int(best_hgbr[2])}")
print(f"  min_samples_leaf = {int(best_hgbr[3])}")
print(f"  Best CV MSE = {score_hgbr:.4f}")

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