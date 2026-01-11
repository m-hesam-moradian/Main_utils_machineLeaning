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
# 2. Spotted Deer Optimization Algorithm (SDOA)
# =====================================================
def sdoa_optimize(objective_func, lb, ub, N=30, T=100, mu=0.1, lambda_=0.5, rho=0.9, theta_div=0.1):
    """
    Spotted Deer Optimization Algorithm (SDOA)
    Inspired by sika deer behaviors:
    - Seasonal migration: spring (global exploration), autumn (local exploitation), burst (diversity recovery)
    - Social status: weighted updates
    - Arena competition: energy-based fighting
    - Fawn learning: perturbation
    - Pheromone regulation: adaptive balance
    """
    start_time = time.time()
    D = len(lb)
    population = lb + np.random.rand(N, D) * (ub - lb)

    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]
    print(f"Iteration 0: Best params = {best_params.round(4)}, Best MSE = {best_score:.4f}")

    # Initialize social status s_i [0,1], energy e_i >0, pheromone P=0.5
    s = np.random.rand(N)
    e = np.random.rand(N) + 1
    P = 0.5

    # Initial diversity
    initial_div = np.mean(np.linalg.norm(population - np.mean(population, axis=0), axis=1))
    theta_div = theta_div * initial_div

    convergence = [best_score]  # For MBE convergence plot

    for t in range(1, T + 1):
        # Compute D_pop
        D_pop = np.mean(np.linalg.norm(population - np.mean(population, axis=0), axis=1))

        # Update P
        P = rho * P + (1 - rho) * (1/N) * np.sum(np.exp(-fitness / (best_score + 1e-10)))

        for i in range(N):
            # Update social status (approximate rank_score as normalized fitness)
            rank_score = 1 - (fitness[i] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-10)
            s[i] = (1 - mu) * s[i] + mu * rank_score

            # Simulate fight and energy update (simple approximation)
            win_flag = 1 if np.random.rand() < s[i] else 0
            e[i] -= 0.1 * (1 - win_flag) + 0.05 * np.random.rand()

            if e[i] < 0:
                e[i] = 0.5  # Reset

            # Position update based on s_i
            if s[i] > 0.5:  # High status: follow best
                new_pos = population[i] + lambda_ * s[i] * (best_params - population[i])
            else:  # Low status: explore
                new_pos = population[i] + (1 - s[i]) * (ub - lb) * (np.random.rand(D) - 0.5) * 0.5

            # Seasonal migration
            if t < T/2:  # Spring: exploration
                new_pos += np.random.rand(D) * (ub - lb) * (1 - t/T)
            else:  # Autumn: exploitation
                new_pos += (1 - t/T) * (best_params - new_pos)

            # Fawn perturbation
            epsilon = 0.1 * (1 - t/T) * np.sin(2 * np.pi * np.random.rand(D))
            new_pos += epsilon

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
                    print(f"Iteration {t:2d} → Best params = {best_params.round(4)}, MSE = {best_score:.4f}")

        # Burst migration if low diversity
        if D_pop < theta_div:
            for i in range(N//2):  # Half the population
                population[i] = lb + np.random.rand(D) * (ub - lb)
                fitness[i] = objective_func(population[i])

        convergence.append(best_score)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nTotal runtime: {runtime:.2f} seconds\n")

    # Convergence plot (MBE = Mean Best Error over iterations)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(convergence)), convergence, label='Best MSE per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Best MSE (MBE)')
    plt.title('Convergence Curve - Spotted Deer Optimization Algorithm')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_params, best_score


# =====================================================
# Objective functions
# =====================================================

# Ridge Regression (RR)
def objective_rr(params):
    alpha = params[0]

    model = Ridge(alpha=alpha)

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Extra Trees Regression (ETR)
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
lb_rr = np.array([0.01])
ub_rr = np.array([100])

lb_etr = np.array([50, 3, 2, 1])
ub_etr = np.array([300, 30, 20, 10])

lb_hgbr = np.array([50, 0.01, 3, 1])
ub_hgbr = np.array([300, 0.3, 15, 20])


# =====================================================
# Run SDOA on all models
# =====================================================

print("\n" + "="*90)
print("SPOTTED DEER OPTIMIZATION ALGORITHM (SDOA) - 2025")
print("="*90)

# 1. Ridge Regression (RR)
print("\nOptimizing Ridge Regression (RR)...")
best_rr, score_rr = sdoa_optimize(objective_rr, lb_rr, ub_rr, N=30, T=80)

print("\nBest RR parameters found:")
print(f"  alpha = {best_rr[0]:.4f}")
print(f"  Best CV MSE = {score_rr:.4f}")

rr_final = Ridge(alpha=best_rr[0])
rr_final.fit(X_train, y_train)
test_mse_rr = mean_squared_error(y_test, rr_final.predict(X_test))
print(f"Final Test MSE (RR) = {test_mse_rr:.4f}\n")


# 2. Extra Trees Regression (ETR)
print("\nOptimizing Extra Trees Regression (ETR)...")
best_etr, score_etr = sdoa_optimize(objective_etr, lb_etr, ub_etr, N=30, T=80)

print("\nBest ETR parameters found:")
print(f"  n_estimators = {int(best_etr[0])}")
print(f"  max_depth = {int(best_etr[1])}")
print(f"  min_samples_split = {int(best_etr[2])}")
print(f"  min_samples_leaf = {int(best_etr[3])}")
print(f"  Best CV MSE = {score_etr:.4f}")

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


# 3. Histogram-Based Gradient Boosting Regression (HGBR)
print("\nOptimizing Histogram-Based Gradient Boosting Regression (HGBR)...")
best_hgbr, score_hgbr = sdoa_optimize(objective_hgbr, lb_hgbr, ub_hgbr, N=30, T=80)

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