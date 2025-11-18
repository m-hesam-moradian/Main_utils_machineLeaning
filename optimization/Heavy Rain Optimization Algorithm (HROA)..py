# --------------------------------------------------------------
#  Heavy Rain Optimization Algorithm (HROA) + QR + SF + LLAR
# --------------------------------------------------------------
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import QuantileRegressor, LassoLars
from sklearn.ensemble import ExtraTreesRegressor
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

# ------------------- 2. Heavy Rain Optimization Algorithm (HROA) -------------------
def hroa_optimize(obj_func, lb, ub, pop_size=30, max_iter=60, verbose=True):
    """
    Heavy Rain Optimization Algorithm (HROA)
    -------------------------------------------------
    Inspired by heavy rain & storm dynamics:
    • Cloud Formation      → Population clustering
    • Raindrop Falling     → Move toward low points (gbest)
    • Lightning Strike     → Lévy-flight global jump
    • Flood Flow           → Directional flow to best
    • Evaporation          → Elite replacement
    -------------------------------------------------
    """
    dim = len(lb)
    # Initialize raindrops
    drops = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in drops])

    gbest_idx = np.argmin(fitness)
    gbest = drops[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best RMSE = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for it in range(1, max_iter + 1):
        t = it / max_iter
        storm_intensity = 1.0 - t  # High early → chaotic rain, late → focused flood

        # Dynamic cloud center (aggregation)
        cloud_center = np.mean(drops, axis=0)

        new_drops = drops.copy()

        for i in range(pop_size):
            r1, r2, r3 = np.random.rand(3)

            # ---- Phase 1: Raindrop Falling (gravity toward best) ----
            gravity_pull = storm_intensity * r1 * (gbest - drops[i])

            # ---- Phase 2: Flood Flow (directional movement) ----
            flood_flow = storm_intensity * r2 * (cloud_center - drops[i])

            candidate = drops[i] + gravity_pull + 0.5 * flood_flow

            # ---- Phase 3: Lightning Strike (exploration) ----
            if np.random.rand() < 0.15 * storm_intensity:
                # Lévy flight simulating lightning
                beta = 1.5
                sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                         (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
                step = 0.01 * (np.random.randn(dim) * sigma) / (np.abs(np.random.randn(dim)) ** (1/beta))
                candidate += step * (ub - lb) * storm_intensity

            # ---- Phase 4: Wind Gust (local turbulence) ----
            wind = (ub - lb) * np.random.randn(dim) * 0.02 * (1 - storm_intensity)
            candidate += wind

            # Clip
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            new_fit = obj_func(candidate)

            # Greedy selection
            if new_fit < fitness[i]:
                new_drops[i] = candidate
                fitness[i] = new_fit
                if new_fit < gbest_fit:
                    gbest = candidate.copy()
                    gbest_fit = new_fit
                    if verbose:
                        print(f"Iter {it:02d} | Best RMSE = {gbest_fit:.5f} | "
                              f"Params = {np.round(gbest, 4)}")

        drops = new_drops

        # ---- Phase 5: Evaporation & New Cloud (elite replacement) ----
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.1:
            drops[worst_idx] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst_idx] = obj_func(drops[worst_idx])

    return gbest, gbest_fit


# ------------------- 3. Objective Functions -------------------
def qr_objective(params):
    quantile = params[0]
    alpha = params[1]
    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver='highs')
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                   cv=5, scoring='neg_mean_squared_error').mean())
    return rmse

def sf_objective(params):
    n_est = int(params[0])
    max_d = int(params[1])
    min_split = int(params[2])
    model = ExtraTreesRegressor(
        n_estimators=n_est,
        max_depth=max_d if max_d > 0 else None,
        min_samples_split=min_split,
        bootstrap=True,
        random_state=42,
        n_jobs=4
    )
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                   cv=5, scoring='neg_mean_squared_error').mean())
    return rmse

def llar_objective(params):
    alpha = params[0]
    max_iter = int(params[1])
    model = LassoLars(alpha=alpha, max_iter=max_iter)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                   cv=5, scoring='neg_mean_squared_error').mean())
    return rmse


# ------------------- 4. Search Bounds -------------------
lb_qr = np.array([0.1, 1e-6])
ub_qr = np.array([0.9, 10.0])

lb_sf = np.array([50, 5, 2])
ub_sf = np.array([500, 50, 20])

lb_llar = np.array([1e-6, 100])
ub_llar = np.array([10.0, 5000])


# ------------------- 5. Run HROA -------------------
print("\n" + "="*70)
print("OPTIMIZING QUANTILE REGRESSION (QR) WITH HROA")
print("="*70)
best_qr, rmse_qr = hroa_optimize(qr_objective, lb_qr, ub_qr, pop_size=25, max_iter=50)

print("\n" + "="*70)
print("OPTIMIZING STOCHASTIC FOREST (SF) WITH HROA")
print("="*70)
best_sf, rmse_sf = hroa_optimize(sf_objective, lb_sf, ub_sf, pop_size=25, max_iter=50)

print("\n" + "="*70)
print("OPTIMIZING LASSO LEAST ANGLE REGRESSION (LLAR) WITH HROA")
print("="*70)
best_llar, rmse_llar = hroa_optimize(llar_objective, lb_llar, ub_llar, pop_size=25, max_iter=50)


# ------------------- 6. Final Evaluation -------------------
# QR
qr_final = QuantileRegressor(quantile=best_qr[0], alpha=best_qr[1], solver='highs')
qr_final.fit(X_train, y_train)
rmse_test_qr = np.sqrt(mean_squared_error(y_test, qr_final.predict(X_test)))

# SF
sf_final = ExtraTreesRegressor(
    n_estimators=int(best_sf[0]),
    max_depth=int(best_sf[1]) if best_sf[1] > 0 else None,
    min_samples_split=int(best_sf[2]),
    bootstrap=True,
    random_state=42,
    n_jobs=4
)
sf_final.fit(X_train, y_train)
rmse_test_sf = np.sqrt(mean_squared_error(y_test, sf_final.predict(X_test)))

# LLAR
llar_final = LassoLars(alpha=best_llar[0], max_iter=int(best_llar[1]))
llar_final.fit(X_train, y_train)
rmse_test_llar = np.sqrt(mean_squared_error(y_test, llar_final.predict(X_test)))


# ------------------- 7. Final Report -------------------
print("\n" + "="*70)
print("FINAL RESULTS (Heavy Rain Optimization Algorithm - HROA)")
print("="*70)
print(f"QR     → quantile={best_qr[0]:.3f}, alpha={best_qr[1]:.2e} | "
      f"CV RMSE={rmse_qr:.4f} | Test RMSE={rmse_test_qr:.4f}")
print(f"SF     → n_est={int(best_sf[0])}, max_depth={int(best_sf[1])}, "
      f"min_split={int(best_sf[2])} | CV RMSE={rmse_sf:.4f} | Test RMSE={rmse_test_sf:.4f}")
print(f"LLAR   → alpha={best_llar[0]:.2e}, max_iter={int(best_llar[1])} | "
      f"CV RMSE={rmse_llar:.4f} | Test RMSE={rmse_test_llar:.4f}")
print("="*70)