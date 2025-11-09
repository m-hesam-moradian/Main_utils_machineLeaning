# --------------------------------------------------------------
#  Kepler Optimization Algorithm (KOA) + RR + KNNR + HGBR
# --------------------------------------------------------------
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ------------------- 1. Load & preprocess California Housing -------------------
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------- 2. Kepler Optimization Algorithm (KOA) -------------------
def koa_optimize(obj_func, lb, ub, pop_size=30, max_iter=50, verbose=True):
    """
    Kepler Optimization Algorithm (KOA)
    -------------------------------------------------
    Inspired by Kepler's laws:
    • Elliptical orbits → position update with gravitational pull
    • Area law → adaptive step size
    • Harmonic law → convergence acceleration
    -------------------------------------------------
    Returns: best_params, best_score (RMSE)
    """
    dim = len(lb)
    # Initialize population (planets)
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])

    gbest_idx = np.argmin(fitness)
    gbest = pop[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best RMSE = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    G = 6.67430e-11  # Gravitational constant (scaled)
    M_sun = 1.0      # Mass of "sun" (best solution)

    for it in range(1, max_iter + 1):
        t = it / max_iter
        a = 2 * (1 - t)  # Linear decreasing inertia

        for i in range(pop_size):
            r = np.random.rand(dim)

            # Distance to global best (sun)
            dist = np.linalg.norm(pop[i] - gbest)
            if dist == 0:
                dist = 1e-8

            # Gravitational force (Kepler-inspired attraction)
            F = G * M_sun / (dist ** 2 + 1e-8)

            # Velocity update (area law)
            velocity = a * (pop[i] - gbest) + F * r * (gbest - pop[i])

            # Position update (elliptical motion)
            candidate = pop[i] + velocity

            # Harmonic law acceleration in late phase
            if t > 0.7:
                candidate += 0.1 * (1 - t) * (gbest - candidate)

            # Boundary handling
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            new_fit = obj_func(candidate)

            # Greedy selection
            if new_fit < fitness[i]:
                pop[i] = candidate
                fitness[i] = new_fit
                if new_fit < gbest_fit:
                    gbest = candidate.copy()
                    gbest_fit = new_fit
                    if verbose:
                        print(f"Iter {it:02d} | Best RMSE = {gbest_fit:.5f} | "
                              f"Params = {np.round(gbest, 4)}")

        # Elite replacement (avoid stagnation)
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.1:
            pop[worst_idx] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst_idx] = obj_func(pop[worst_idx])

    return gbest, gbest_fit


# ------------------- 3. Objective Functions -------------------
def rr_objective(params):
    alpha = params[0]
    model = Ridge(alpha=alpha)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                   cv=5, scoring='neg_mean_squared_error').mean())
    return rmse

def knnr_objective(params):
    n_neighbors = int(params[0])
    weights_idx = int(params[1])  # 0: uniform, 1: distance
    p = int(params[2])
    weights = ['uniform', 'distance'][weights_idx]
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                   cv=5, scoring='neg_mean_squared_error').mean())
    return rmse

def hgbr_objective(params):
    max_iter = int(params[0])
    lr = params[1]
    max_depth = int(params[2])
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=lr,
        max_depth=max_depth if max_depth > 0 else None,
        random_state=42
    )
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                   cv=5, scoring='neg_mean_squared_error').mean())
    return rmse


# ------------------- 4. Search Bounds -------------------
lb_rr = np.array([0.01])
ub_rr = np.array([100.0])

lb_knnr = np.array([1, 0, 1])        # n_neighbors, weights_idx, p
ub_knnr = np.array([50, 1, 2])

lb_hgbr = np.array([50, 0.01, 3])     # max_iter, lr, max_depth
ub_hgbr = np.array([300, 0.3, 30])


# ------------------- 5. Run KOA -------------------
print("\n" + "="*70)
print("OPTIMIZING RIDGE REGRESSION (RR) WITH KOA")
print("="*70)
best_rr, rmse_rr = koa_optimize(rr_objective, lb_rr, ub_rr, pop_size=20, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING KNN REGRESSION (KNNR) WITH KOA")
print("="*70)
best_knnr, rmse_knnr = koa_optimize(knnr_objective, lb_knnr, ub_knnr, pop_size=20, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING HGB REGRESSION (HGBR) WITH KOA")
print("="*70)
best_hgbr, rmse_hgbr = koa_optimize(hgbr_objective, lb_hgbr, ub_hgbr, pop_size=20, max_iter=40)


# ------------------- 6. Final Evaluation -------------------
# RR
rr_final = Ridge(alpha=best_rr[0])
rr_final.fit(X_train, y_train)
rmse_test_rr = np.sqrt(mean_squared_error(y_test, rr_final.predict(X_test)))

# KNNR
weights = ['uniform', 'distance'][int(best_knnr[1])]
knnr_final = KNeighborsRegressor(
    n_neighbors=int(best_knnr[0]),
    weights=weights,
    p=int(best_knnr[2])
)
knnr_final.fit(X_train, y_train)
rmse_test_knnr = np.sqrt(mean_squared_error(y_test, knnr_final.predict(X_test)))

# HGBR
hgbr_final = HistGradientBoostingRegressor(
    max_iter=int(best_hgbr[0]),
    learning_rate=best_hgbr[1],
    max_depth=int(best_hgbr[2]) if best_hgbr[2] > 0 else None,
    random_state=42
)
hgbr_final.fit(X_train, y_train)
rmse_test_hgbr = np.sqrt(mean_squared_error(y_test, hgbr_final.predict(X_test)))


# ------------------- 7. Final Report -------------------
print("\n" + "="*70)
print("FINAL RESULTS (Kepler Optimization Algorithm - KOA)")
print("="*70)
print(f"RR     → alpha={best_rr[0]:.4f} | CV RMSE={rmse_rr:.4f} | Test RMSE={rmse_test_rr:.4f}")
print(f"KNNR   → n_neighbors={int(best_knnr[0])}, weights={weights}, p={int(best_knnr[2])} | "
      f"CV RMSE={rmse_knnr:.4f} | Test RMSE={rmse_test_knnr:.4f}")
print(f"HGBR   → max_iter={int(best_hgbr[0])}, lr={best_hgbr[1]:.4f}, depth={int(best_hgbr[2])} | "
      f"CV RMSE={rmse_hgbr:.4f} | Test RMSE={rmse_test_hgbr:.4f}")
print("="*70)