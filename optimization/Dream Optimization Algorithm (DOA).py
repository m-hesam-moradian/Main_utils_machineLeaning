# --------------------------------------------------------------
#  Dream Optimization Algorithm (DOA) + QR + SF + LLAR
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

# Scale features (important for QR and LassoLars)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- 2. Dream Optimization Algorithm (DOA) -------------------
def doa_optimize(obj_func, lb, ub, pop_size=25, max_iter=50, verbose=True):
    """
    Dream Optimization Algorithm (DOA)
    -------------------------------------------------
    Inspired by human dreaming:
    • Lucid Dreaming       → Large creative jumps (exploration)
    • REM Sleep            → Move toward best dream (exploitation)
    • Subconscious Refinement → Small Gaussian polish
    • Dream Recall         → Elite replacement
    -------------------------------------------------
    """
    dim = len(lb)
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])

    gbest_idx = np.argmin(fitness)
    gbest = pop[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best RMSE = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for it in range(1, max_iter + 1):
        dream_factor = 1.0 - it / max_iter  # High early = creative chaos

        for i in range(pop_size):
            r1 = np.random.rand(dim)

            # ---- Lucid Dreaming: Creative leaps ----
            if np.random.rand() < dream_factor:
                step = (ub - lb) * (np.random.rand(dim) - 0.5) * 2.0 * dream_factor
                candidate = pop[i] + step
            else:
                # ---- REM Sleep: Move to best dream ----
                candidate = pop[i] + r1 * (gbest - pop[i]) * (1 - dream_factor)

            # ---- Subconscious Refinement ----
            noise = (ub - lb) * np.random.randn(dim) * 0.03 * (1 - dream_factor)
            candidate += noise

            # Clip
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            new_fit = obj_func(candidate)

            # Greedy update
            if new_fit < fitness[i]:
                pop[i] = candidate
                fitness[i] = new_fit
                if new_fit < gbest_fit:
                    gbest = candidate.copy()
                    gbest_fit = new_fit
                    if verbose:
                        print(f"Iter {it:02d} | Best RMSE = {gbest_fit:.5f} | "
                              f"Params = {np.round(gbest, 4)}")

        # ---- Dream Recall: Replace worst dream ----
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.1:
            pop[worst_idx] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst_idx] = obj_func(pop[worst_idx])

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
        bootstrap=True,        # Makes it stochastic
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


# ------------------- 5. Run DOA -------------------
print("\n" + "="*70)
print("OPTIMIZING QUANTILE REGRESSION (QR) WITH DOA")
print("="*70)
best_qr, rmse_qr = doa_optimize(qr_objective, lb_qr, ub_qr, pop_size=20, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING STOCHASTIC FOREST (SF) WITH DOA")
print("="*70)
best_sf, rmse_sf = doa_optimize(sf_objective, lb_sf, ub_sf, pop_size=20, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING LASSO LEAST ANGLE REGRESSION (LLAR) WITH DOA")
print("="*70)
best_llar, rmse_llar = doa_optimize(llar_objective, lb_llar, ub_llar, pop_size=20, max_iter=40)


# ------------------- 6. Final Evaluation -------------------
# QR
qr_final = QuantileRegressor(quantile=best_qr[0], alpha=best_qr[1], solver='highs')
qr_final.fit(X_train, y_train)
rmse_test_qr = np.sqrt(mean_squared_error(y_test, qr_final.predict(X_test)))

# SF (Stochastic Forest)
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
print("FINAL RESULTS (Dream Optimization Algorithm - DOA)")
print("="*70)
print(f"QR     → quantile={best_qr[0]:.3f}, alpha={best_qr[1]:.2e} | "
      f"CV RMSE={rmse_qr:.4f} | Test RMSE={rmse_test_qr:.4f}")
print(f"SF     → n_est={int(best_sf[0])}, max_depth={int(best_sf[1])}, "
      f"min_split={int(best_sf[2])} | CV RMSE={rmse_sf:.4f} | Test RMSE={rmse_test_sf:.4f}")
print(f"LLAR   → alpha={best_llar[0]:.2e}, max_iter={int(best_llar[1])} | "
      f"CV RMSE={rmse_llar:.4f} | Test RMSE={rmse_test_llar:.4f}")
print("="*70)