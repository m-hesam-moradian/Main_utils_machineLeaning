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
# 2. Orangutan Optimization Algorithm (OOA)
# =====================================================
def ooa_optimize(objective_func, lb, ub, N=25, T=60):
    """
    Orangutan Optimization Algorithm (OOA)
    Inspired by orangutan foraging and nesting behaviors:
    - Foraging: exploration toward better solutions
    - Nesting: local exploitation with adaptive scaling
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

    convergence = [best_score]  # For convergence plot

    for t in range(1, T + 1):
        for i in range(N):
            # Phase 1: Foraging Strategy (Exploration)
            better_indices = np.where(fitness < fitness[i])[0]
            if len(better_indices) > 0:
                food_idx = np.random.choice(better_indices)
                food_pos = population[food_idx]
            else:
                food_pos = best_params  # Use best if no better

            r = np.random.rand(D)
            I = np.random.choice([1, 2], size=D)
            new_pos_p1 = population[i] + r * (food_pos - I * population[i])

            # Clip to bounds
            new_pos_p1 = np.clip(new_pos_p1, lb, ub)

            # Evaluate and update if better (greedy)
            new_fit_p1 = objective_func(new_pos_p1)
            if new_fit_p1 < fitness[i]:
                population[i] = new_pos_p1
                fitness[i] = new_fit_p1
                if new_fit_p1 < best_score:
                    best_params = new_pos_p1.copy()
                    best_score = new_fit_p1
                    print(f"Iteration {t:2d} → Best params = {best_params.round(4)}, MSE = {best_score:.4f}")

            # Phase 2: Nesting Skill (Exploitation)
            new_pos_p2 = population[i] + (1 - 2 * np.random.rand(D) / t) * (ub - lb)

            # Clip to bounds
            new_pos_p2 = np.clip(new_pos_p2, lb, ub)

            # Evaluate and update if better (greedy)
            new_fit_p2 = objective_func(new_pos_p2)
            if new_fit_p2 < fitness[i]:
                population[i] = new_pos_p2
                fitness[i] = new_fit_p2
                if new_fit_p2 < best_score:
                    best_params = new_pos_p2.copy()
                    best_score = new_fit_p2
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
    plt.title('Convergence Curve - Orangutan Optimization Algorithm')
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
# Run OOA on all models
# =====================================================

print("\n" + "="*90)
print("ORANGUTAN OPTIMIZATION ALGORITHM (OOA) - 2024")
print("="*90)

# 1. Ridge Regression
print("\nOptimizing Ridge Regression (RR)...")
best_rr, score_rr = ooa_optimize(objective_rr, lb_rr, ub_rr, N=25, T=60)

print("\nBest RR parameters found:")
print(f"  alpha = {best_rr[0]:.4f}")
print(f"  Best CV MSE = {score_rr:.4f}")

rr_final = Ridge(alpha=best_rr[0])
rr_final.fit(X_train, y_train)
test_mse_rr = mean_squared_error(y_test, rr_final.predict(X_test))
print(f"Final Test MSE (RR) = {test_mse_rr:.4f}\n")


# 2. Extra Trees Regression
print("\nOptimizing Extra Trees Regression (ETR)...")
best_etr, score_etr = ooa_optimize(objective_etr, lb_etr, ub_etr, N=25, T=60)

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


# 3. Histogram-Based Gradient Boosting Regression
print("\nOptimizing Histogram-Based Gradient Boosting Regression (HGBR)...")
best_hgbr, score_hgbr = ooa_optimize(objective_hgbr, lb_hgbr, ub_hgbr, N=25, T=80)

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