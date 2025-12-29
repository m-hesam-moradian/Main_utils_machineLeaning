import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# =====================================================
# 1. Load your local Excel dataset
# =====================================================
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

print(f"Loading dataset from: {file_path}")
df = pd.read_excel(file_path)

# Assuming:
# - Last column is the target (continuous for regression)
# - All other columns are features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Dataset loaded → Features shape: {X.shape}, Target shape: {y.shape}")

# Scale features (highly recommended for most models)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================================
# 2. Red Panda Optimization Algorithm (RPOA)
# =====================================================
def rpoa_optimize(objective_func, lb, ub, N=25, T=80):
    """
    Red Panda Optimization Algorithm (RPOA)
    - Foraging: random jumps (exploration)
    - Climbing: directional movement toward best (exploitation)
    - Resting: local Gaussian refinement
    - Jumping: occasional big leaps (diversification)
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
    print(f"Iteration 0: Best params = {best_params}, Best MSE = {best_score:.4f}")

    for t in range(1, T + 1):
        for i in range(N):
            # Phase 1: Foraging (Exploration)
            if np.random.rand() < 0.55:  # ~55% chance → foraging
                step = (ub - lb) * np.random.uniform(0.08, 0.35, D)
                direction = np.sign(np.random.randn(D))
                new_pos = population[i] + direction * np.random.rand(D) * step
            else:
                # Phase 2: Climbing (Exploitation)
                r = np.random.rand(D)
                climb_factor = (1 - t / T) * r
                new_pos = population[i] + climb_factor * (best_params - population[i])

            # Phase 3: Resting (Local refinement)
            rest = (ub - lb) * np.random.randn(D) * (1 / (t + 5))
            new_pos += rest

            # Phase 4: Jumping (Diversification)
            if np.random.rand() < 0.12 * (1 - t / T):
                jump_strength = (ub - lb) * np.random.uniform(0.06, 0.25, D)
                new_pos += jump_strength * np.random.randn(D)

            # Clip
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)

            # Greedy update
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t:2d} → Best params = {best_params.round(4)}, MSE = {best_score:.4f}")

        # Occasional elite replacement (refresh worst)
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.06:
            population[worst_idx] = lb + np.random.rand(D) * (ub - lb)
            fitness[worst_idx] = objective_func(population[worst_idx])

    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nTotal runtime: {runtime:.2f} seconds\n")

    return best_params, best_score


# =====================================================
# Objective functions
# =====================================================

# 1. Histogram-Based Gradient Boosting Regression (HGBR)
def objective_hgbr(params):
    mi = int(params[0])
    lr = params[1]
    md = int(params[2])
    msl = int(params[3])

    model = HistGradientBoostingRegressor(
        max_iter=mi,
        learning_rate=lr,
        max_depth=md,
        min_samples_leaf=msl,
        random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score  # We minimize MSE → return negative


# 2. Decision Tree Regression (DTR)
def objective_dtr(params):
    md = int(params[0])
    mss = int(params[1])
    msl = int(params[2])

    model = DecisionTreeRegressor(
        max_depth=md,
        min_samples_split=mss,
        min_samples_leaf=msl,
        random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# =====================================================
# Hyperparameter search bounds
# =====================================================

# HGBR bounds
lb_hgbr = np.array([50, 0.01, 3, 1])
ub_hgbr = np.array([400, 0.3, 20, 25])

# DTR bounds
lb_dtr = np.array([3, 2, 1])
ub_dtr = np.array([25, 25, 12])


# =====================================================
# Run RPOA on both models
# =====================================================

print("\n" + "="*80)
print("RED PANDA OPTIMIZATION ALGORITHM (RPOA) - 2025")
print("="*80)

# 1. HGBR
print("\nOptimizing Histogram-Based Gradient Boosting Regression (HGBR)...")
best_hgbr, score_hgbr = rpoa_optimize(objective_hgbr, lb_hgbr, ub_hgbr, N=25, T=60)

print("\nBest HGBR parameters found:")
print(f"  max_iter         = {int(best_hgbr[0])}")
print(f"  learning_rate    = {best_hgbr[1]:.4f}")
print(f"  max_depth        = {int(best_hgbr[2])}")
print(f"  min_samples_leaf = {int(best_hgbr[3])}")
print(f"  Best CV MSE      = {score_hgbr:.4f}")

# Final model evaluation
hgbr_final = HistGradientBoostingRegressor(
    max_iter=int(best_hgbr[0]),
    learning_rate=best_hgbr[1],
    max_depth=int(best_hgbr[2]),
    min_samples_leaf=int(best_hgbr[3]),
    random_state=42
)
hgbr_final.fit(X_train, y_train)
y_pred_hgbr = hgbr_final.predict(X_test)
test_mse_hgbr = mean_squared_error(y_test, y_pred_hgbr)
print(f"Final Test MSE (HGBR) = {test_mse_hgbr:.4f}\n")


# 2. DTR
print("\nOptimizing Decision Tree Regression (DTR)...")
best_dtr, score_dtr = rpoa_optimize(objective_dtr, lb_dtr, ub_dtr, N=25, T=60)

print("\nBest DTR parameters found:")
print(f"  max_depth         = {int(best_dtr[0])}")
print(f"  min_samples_split = {int(best_dtr[1])}")
print(f"  min_samples_leaf  = {int(best_dtr[2])}")
print(f"  Best CV MSE       = {score_dtr:.4f}")

# Final model evaluation
dtr_final = DecisionTreeRegressor(
    max_depth=int(best_dtr[0]),
    min_samples_split=int(best_dtr[1]),
    min_samples_leaf=int(best_dtr[2]),
    random_state=42
)
dtr_final.fit(X_train, y_train)
y_pred_dtr = dtr_final.predict(X_test)
test_mse_dtr = mean_squared_error(y_test, y_pred_dtr)
print(f"Final Test MSE (DTR) = {test_mse_dtr:.4f}\n")