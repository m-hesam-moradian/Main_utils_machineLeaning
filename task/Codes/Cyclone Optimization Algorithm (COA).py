# --------------------------------------------------------------
#  Cyclone Optimization Algorithm (COA) + RR + KNNR + CATR
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

# ------------------- 1. Load & preprocess -------------------
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_s, X_test_s = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)

# ------------------- 2. Cyclone Optimization Algorithm (COA) -------------------
def coa_optimize(obj_func, lb, ub, pop_size=30, max_iter=60, verbose=True):
    """
    Cyclone Optimization Algorithm (COA)
    -------------------------------------------------
    Inspired by tropical cyclone dynamics:
    • Formation & Intensification → Spiral movement + pressure drop
    • Eye of the Storm           → Calm exploitation zone
    • Spiral Rainbands           → Lévy-like jumps
    • Landfall Decay             → Adaptive convergence
    • Warm Core                  → Elite memory
    -------------------------------------------------
    """
    dim = len(lb)
    storms = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(s) for s in storms])

    gbest_idx = np.argmin(fitness)
    gbest = storms[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best RMSE = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for it in range(1, max_iter + 1):
        t = it / max_iter
        pressure = 1.0 - t  # High pressure early → chaos, low late → calm eye

        # Eye of the cyclone (calm center from top 20%)
        eye_ratio = 0.2
        elite_count = max(1, int(eye_ratio * pop_size))
        elite_idx = np.argsort(fitness)[:elite_count]
        eye_center = np.mean(storms[elite_idx], axis=0)

        new_storms = np.zeros_like(storms)

        for i in range(pop_size):
            r1, r2, r3 = np.random.rand(3)

            # ---- Spiral Rainbands (Lévy-like chaos) ----
            if r1 < pressure:
                angle = 2 * np.pi * np.random.rand()
                radius = pressure * np.random.pareto(1.5)
                spiral_step = radius * np.array([np.cos(angle), np.sin(angle)])
                if dim > 2:
                    spiral_step = np.tile(spiral_step, (dim + 1) // 2)[:dim]
                candidate = storms[i] + spiral_step * (ub - lb) * 0.1
            else:
                # ---- Inflow toward Eye (exploitation) ----
                to_eye = eye_center - storms[i]
                to_gbest = gbest - storms[i]
                candidate = storms[i] + pressure * (r2 * to_eye + (1 - r2) * to_gbest)

            # ---- Warm Core Updraft (small refinement) ----
            updraft = (ub - lb) * np.random.randn(dim) * 0.02 * (1 - pressure)
            candidate += updraft

            # ---- Landfall Decay (reduce step size near end) ----
            if t > 0.8:
                candidate = 0.7 * candidate + 0.3 * gbest

            candidate = np.clip(candidate, lb, ub)
            new_fit = obj_func(candidate)

            # Greedy selection
            if new_fit < fitness[i]:
                new_storms[i] = candidate
                fitness[i] = new_fit
                if new_fit < gbest_fit:
                    gbest = candidate.copy()
                    gbest_fit = new_fit
                    if verbose:
                        print(f"Iter {it:02d} | Best RMSE = {gbest_fit:.5f} | "
                              f"Params = {np.round(gbest, 4)}")
            else:
                new_storms[i] = storms[i]

        storms = new_storms

        # ---- New Depression Formation (replace worst) ----
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.12:
            storms[worst_idx] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst_idx] = obj_func(storms[worst_idx])

    return gbest, gbest_fit


# ------------------- 3. Objective Functions -------------------
def rr_objective(params):
    model = Ridge(alpha=params[0])
    return np.sqrt(-cross_val_score(model, X_train_s, y_train, cv=5,
                                   scoring='neg_mean_squared_error').mean())

def knnr_objective(params):
    n = int(params[0])
    w_idx = int(params[1])
    p = int(params[2])
    weights = ['uniform', 'distance'][w_idx]
    model = KNeighborsRegressor(n_neighbors=n, weights=weights, p=p)
    return np.sqrt(-cross_val_score(model, X_train_s, y_train, cv=5,
                                   scoring='neg_mean_squared_error').mean())

def catr_objective(params):
    depth = int(params[0])
    lr = params[1]
    iters = int(params[2])
    l2 = params[3]
    model = CatBoostRegressor(depth=depth, learning_rate=lr, iterations=iters,
                              l2_leaf_reg=l2, random_seed=42, verbose=False)
    return np.sqrt(-cross_val_score(model, X_train, y_train, cv=5,
                                   scoring='neg_mean_squared_error').mean())


# ------------------- 4. Bounds -------------------
lb_rr = np.array([0.01])
ub_rr = np.array([200.0])

lb_knnr = np.array([1, 0, 1])
ub_knnr = np.array([60, 1, 2])

lb_catr = np.array([4, 0.01, 100, 0.1])
ub_catr = np.array([12, 0.3, 1500, 20.0])


# ------------------- 5. Run COA -------------------
print("\n" + "="*80)
print("OPTIMIZING RIDGE REGRESSION (RR) WITH CYCLONE OPTIMIZATION ALGORITHM (COA)")
print("="*80)
best_rr, rmse_rr = coa_optimize(rr_objective, lb_rr, ub_rr, pop_size=30, max_iter=50)

print("\n" + "="*80)
print("OPTIMIZING KNN REGRESSION (KNNR) WITH COA")
print("="*80)
best_knnr, rmse_knnr = coa_optimize(knnr_objective, lb_knnr, ub_knnr, pop_size=30, max_iter=50)

print("\n" + "="*80)
print("OPTIMIZING CATBOOST REGRESSION (CATR) WITH COA")
print("="*80)
best_catr, rmse_catr = coa_optimize(catr_objective, lb_catr, ub_catr, pop_size=30, max_iter=50)


# ------------------- 6. Final Evaluation -------------------
rr_final = Ridge(alpha=best_rr[0])
rr_final.fit(X_train_s, y_train)
test_rmse_rr = np.sqrt(mean_squared_error(y_test, rr_final.predict(X_test_s)))

weights_str = ['uniform', 'distance'][int(best_knnr[1])]
knnr_final = KNeighborsRegressor(n_neighbors=int(best_knnr[0]), weights=weights_str, p=int(best_knnr[2]))
knnr_final.fit(X_train_s, y_train)
test_rmse_knnr = np.sqrt(mean_squared_error(y_test, knnr_final.predict(X_test_s)))

catr_final = CatBoostRegressor(
    depth=int(best_catr[0]), learning_rate=best_catr[1],
    iterations=int(best_catr[2]), l2_leaf_reg=best_catr[3],
    random_seed=42, verbose=False
)
catr_final.fit(X_train, y_train)
test_rmse_catr = np.sqrt(mean_squared_error(y_test, catr_final.predict(X_test)))


# ------------------- 7. Final Report -------------------
print("\n" + "="*80)
print("FINAL RESULTS (Cyclone Optimization Algorithm - COA)")
print("="*80)
print(f"RR     → alpha={best_rr[0]:.4f} | CV RMSE={rmse_rr:.4f} | Test RMSE={test_rmse_rr:.4f}")
print(f"KNNR   → n_neighbors={int(best_knnr[0])}, weights={weights_str}, p={int(best_knnr[2])} | "
      f"CV RMSE={rmse_knnr:.4f} | Test RMSE={test_rmse_knnr:.4f}")
print(f"CATR   → depth={int(best_catr[0])}, lr={best_catr[1]:.4f}, "
      f"iters={int(best_catr[2])}, l2={best_catr[3]:.3f} | "
      f"CV RMSE={rmse_catr:.4f} | Test RMSE={test_rmse_catr:.4f}")
print("="*80)