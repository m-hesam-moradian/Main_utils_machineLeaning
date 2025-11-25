# --------------------------------------------------------------
#  Spider Wasp Optimization (SWO) + RR + KNNR + CATR
# --------------------------------------------------------------
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# ------------------- 1. Load & preprocess California Housing -------------------
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

# Standard scale for RR and KNNR (CatBoost handles raw data well)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

# ------------------- 2. Spider Wasp Optimization (SWO) -------------------
def swo_optimize(obj_func, lb, ub, pop_size=30, max_iter=60, verbose=True):
    """
    Spider Wasp Optimization (SWO)
    -------------------------------------------------
    Inspired by spider wasp hunting strategy:
    • Aerial Search           → Lévy flight (exploration)
    • Target Detection        → Move toward prey (gbest)
    • Paralyzing Sting        → Aggressive exploitation
    • Nest Drag               → Directional pull from elite
    • Venom Effect            → Adaptive step size decay
    -------------------------------------------------
    """
    dim = len(lb)
    # Initialize wasp population
    wasps = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(w) for w in wasps])

    gbest_idx = np.argmin(fitness)
    gbest = wasps[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best RMSE = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for it in range(1, max_iter + 1):
        venom = 1.0 - it / max_iter  # Venom weakens → less chaos, more precision

        elite_ratio = 0.2
        elite_count = max(1, int(elite_ratio * pop_size))
        elite_idx = np.argsort(fitness)[:elite_count]
        elite_center = np.mean(wasps[elite_idx], axis=0)

        new_wasps = wasps.copy()

        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()

            # ---- Phase 1: Aerial Search (Lévy flight) ----
            if r1 < venom:
                beta = 1.5
                sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                         (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
                step = 0.01 * np.random.randn(dim) * sigma / (np.abs(np.random.randn(dim)) ** (1/beta))
                candidate = wasps[i] + step * (ub - lb) * venom
            else:
                # ---- Phase 2: Target Detection & Sting ----
                to_prey = gbest - wasps[i]
                to_nest = elite_center - wasps[i]
                candidate = wasps[i] + venom * (r2 * to_prey + (1 - r2) * to_nest)

            # ---- Phase 3: Paralyzing Venom (local refinement) ----
            sting = (ub - lb) * np.random.randn(dim) * 0.02 * (1 - venom)
            candidate += sting

            # Clip
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            new_fit = obj_func(candidate)

            # Greedy selection
            if new_fit < fitness[i]:
                new_wasps[i] = candidate
                fitness[i] = new_fit
                if new_fit < gbest_fit:
                    gbest = candidate.copy()
                    gbest_fit = new_fit
                    if verbose:
                        print(f"Iter {it:02d} | Best RMSE = {gbest_fit:.5f} | "
                              f"Params = {np.round(gbest, 4)}")

        wasps = new_wasps

        # ---- Phase 4: Failed Hunt → New Territory (elite replacement) ----
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.1:
            wasps[worst_idx] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst_idx] = obj_func(wasps[worst_idx])

    return gbest, gbest_fit


# ------------------- 3. Objective Functions -------------------
def rr_objective(params):
    alpha = params[0]
    model = Ridge(alpha=alpha)
    rmse = np.sqrt(-cross_val_score(model, X_train_scaled, y_train,
                                   cv=5, scoring='neg_mean_squared_error').mean())
    return rmse

def knnr_objective(params):
    n_neighbors = int(params[0])
    weights_idx = int(params[1])
    p = int(params[2])
    weights = ['uniform', 'distance'][weights_idx]
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    rmse = np.sqrt(-cross_val_score(model, X_train_scaled, y_train,
                                   cv=5, scoring='neg_mean_squared_error').mean())
    return rmse

def catr_objective(params):
    depth = int(params[0])
    lr = params[1]
    iterations = int(params[2])
    l2 = params[3]
    model = CatBoostRegressor(
        depth=depth,
        learning_rate=lr,
        iterations=iterations,
        l2_leaf_reg=l2,
        random_seed=42,
        verbose=False,
        thread_count=4
    )
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                   cv=5, scoring='neg_mean_squared_error').mean())
    return rmse


# ------------------- 4. Search Bounds -------------------
lb_rr = np.array([0.01])
ub_rr = np.array([100.0])

lb_knnr = np.array([1, 0, 1])
ub_knnr = np.array([50, 1, 2])

lb_catr = np.array([4, 0.01, 100, 1.0])
ub_catr = np.array([12, 0.3, 1000, 10.0])


# ------------------- 5. Run SWO -------------------
print("\n" + "="*70)
print("OPTIMIZING RIDGE REGRESSION (RR) WITH SPIDER WASP OPTIMIZER (SWO)")
print("="*70)
best_rr, rmse_rr = swo_optimize(rr_objective, lb_rr, ub_rr, pop_size=25, max_iter=50)

print("\n" + "="*70)
print("OPTIMIZING KNN REGRESSION (KNNR) WITH SWO")
print("="*70)
best_knnr, rmse_knnr = swo_optimize(knnr_objective, lb_knnr, ub_knnr, pop_size=25, max_iter=50)

print("\n" + "="*70)
print("OPTIMIZING CATBOOST REGRESSION (CATR) WITH SWO")
print("="*70)
best_catr, rmse_catr = swo_optimize(catr_objective, lb_catr, ub_catr, pop_size=25, max_iter=50)


# ------------------- 6. Final Evaluation -------------------
# RR
rr_final = Ridge(alpha=best_rr[0])
rr_final.fit(X_train_scaled, y_train)
rmse_test_rr = np.sqrt(mean_squared_error(y_test, rr_final.predict(X_test_scaled)))

# KNNR
weights_str = ['uniform', 'distance'][int(best_knnr[1])]
knnr_final = KNeighborsRegressor(
    n_neighbors=int(best_knnr[0]),
    weights=weights_str,
    p=int(best_knnr[2])
)
knnr_final.fit(X_train_scaled, y_train)
rmse_test_knnr = np.sqrt(mean_squared_error(y_test, knnr_final.predict(X_test_scaled)))

# CATR
catr_final = CatBoostRegressor(
    depth=int(best_catr[0]),
    learning_rate=best_catr[1],
    iterations=int(best_catr[2]),
    l2_leaf_reg=best_catr[3],
    random_seed=42,
    verbose=False
)
catr_final.fit(X_train, y_train)
rmse_test_catr = np.sqrt(mean_squared_error(y_test, catr_final.predict(X_test)))


# ------------------- 7. Final Report -------------------
print("\n" + "="*70)
print("FINAL RESULTS (Spider Wasp Optimization - SWO)")
print("="*70)
print(f"RR     → alpha={best_rr[0]:.4f} | CV RMSE={rmse_rr:.4f} | Test RMSE={rmse_test_rr:.4f}")
print(f"KNNR   → n_neighbors={int(best_knnr[0])}, weights={weights_str}, p={int(best_knnr[2])} | "
      f"CV RMSE={rmse_knnr:.4f} | Test RMSE={rmse_test_knnr:.4f}")
print(f"CATR   → depth={int(best_catr[0])}, lr={best_catr[1]:.4f}, "
      f"iter={int(best_catr[2])}, l2={best_catr[3]:.2f} | "
      f"CV RMSE={rmse_catr:.4f} | Test RMSE={rmse_test_catr:.4f}")
print("="*70)