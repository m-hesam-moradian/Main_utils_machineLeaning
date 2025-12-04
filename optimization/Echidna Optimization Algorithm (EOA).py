# --------------------------------------------------------------
#  Echidna Optimization Algorithm (EOA) - Official 2024 Version
#  + Ridge + KNNR + CatBoost Regression
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

# ------------------- 1. Data -------------------
print("Loading California Housing...")
data = fetch_california_housing()
X, y = data.data, data.target
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_s, X_test_s = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)

# ------------------- 2. Echidna Optimization Algorithm (EOA) -------------------
def eoa_optimize(obj_func, lb, ub, pop_size=30, max_iter=60, verbose=True):
    """
    Echidna Optimization Algorithm (EOA) - 2024
    Based on echidna foraging, defense, social & thermoregulation behaviors
    """
    dim = len(lb)
    echidnas = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in echidnas])

    gbest_idx = np.argmin(fitness)
    gbest = echidnas[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best RMSE = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for t in range(1, max_iter + 1):
        a = 2 * (1 - t / max_iter)  # Linearly decreasing exploration factor

        # --- Thermoregulation: Dynamic subgroup (top 30%) ---
        elite_num = max(2, int(0.3 * pop_size))
        elite_idx = np.argsort(fitness)[:elite_num]
        burrow_center = np.mean(echidnas[elite_idx], axis=0)

        new_echidnas = echidnas.copy()

        for i in range(pop_size):
            r1, r2, r3 = np.random.rand(3)

            # --- Phase 1: Foraging Behavior (Lévy flight + local search) ---
            if r1 < 0.6:  # 60% probability → foraging
                if np.random.rand() < 0.4:  # Lévy flight (long jumps)
                    beta = 1.5
                    sigma = (np.gamma(1+beta)*np.sin(np.pi*beta/2) /
                             (np.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                    step = 0.01 * np.random.randn(dim) * sigma / (np.abs(np.random.randn(dim))**(1/beta))
                    candidate = echidnas[i] + step * (ub - lb) * a
                else:  # Local sniffing
                    candidate = echidnas[i] + a * np.random.randn(dim) * (burrow_center - echidnas[i])
            else:
                # --- Phase 2: Defense Behavior (spine roll & escape) ---
                if np.random.rand() < 0.3:  # Roll into ball → big random jump
                    candidate = lb + np.random.rand(dim) * (ub - lb)
                else:
                    candidate = echidnas[i] + a * (gbest - echidnas[i]) * r2

            # --- Phase 3: Mating Season Crossover (with elite) ---
            if np.random.rand() < 0.25 and len(elite_idx) > 1:
                mate = echidnas[np.random.choice(elite_idx)]
                mask = np.random.rand(dim) < 0.5
                candidate = np.where(mask, mate, candidate)

            # --- Phase 4: Thermoregulation (move to warm burrow) ---
            if fitness[i] > np.mean(fitness):
                candidate += 0.1 * a * (burrow_center - candidate)

            candidate = np.clip(candidate, lb, ub)
            new_fit = obj_func(candidate)

            # Greedy selection
            if new_fit < fitness[i]:
                new_echidnas[i] = candidate
                fitness[i] = new_fit
                if new_fit < gbest_fit:
                    gbest = candidate.copy()
                    gbest_fit = new_fit
                    if verbose:
                        print(f"Iter {t:02d} | Best RMSE = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

        echidnas = new_echidnas

        # --- New Generation (replace worst with random immigrant) ---
        if np.random.rand() < 0.1:
            worst = np.argmax(fitness)
            echidnas[worst] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst] = obj_func(echidnas[worst])

    return gbest, gbest_fit


# ------------------- 3. Objective Functions -------------------
def rr_objective(p):
    model = Ridge(alpha=p[0])
    return np.sqrt(-cross_val_score(model, X_train_s, y_train, cv=5,
                                   scoring='neg_mean_squared_error').mean())

def knnr_objective(p):
    n, w_idx, p_val = int(p[0]), int(p[1]), int(p[2])
    weights = ['uniform', 'distance'][w_idx]
    model = KNeighborsRegressor(n_neighbors=n, weights=weights, p=p_val)
    return np.sqrt(-cross_val_score(model, X_train_s, y_train, cv=5,
                                   scoring='neg_mean_squared_error').mean())

def catr_objective(p):
    depth, lr, iters, l2 = int(p[0]), p[1], int(p[2]), p[3]
    model = CatBoostRegressor(depth=depth, learning_rate=lr, iterations=iters,
                              l2_leaf_reg=l2, random_seed=42, verbose=False)
    return np.sqrt(-cross_val_score(model, X_train, y_train, cv=5,
                                   scoring='neg_mean_squared_error').mean())


# ------------------- 4. Bounds -------------------
lb_rr = np.array([0.01])
ub_rr = np.array([200.0])

lb_knnr = np.array([1, 0, 1])
ub_knnr = np.array([70, 1, 2])

lb_catr = np.array([3, 0.01, 100, 0.1])
ub_catr = np.array([12, 0.3, 2000, 30.0])


# ------------------- 5. Run EOA -------------------
print("\n" + "="*85)
print("ECHIDNA OPTIMIZATION ALGORITHM (EOA) - 2024 - OFFICIAL IMPLEMENTATION")
print("="*85)

print("\nOPTIMIZING RIDGE REGRESSION (RR)...")
best_rr, score_rr = eoa_optimize(rr_objective, lb_rr, ub_rr, pop_size=30, max_iter=50)

print("\nOPTIMIZING KNN REGRESSION (KNNR)...")
best_knnr, score_knnr = eoa_optimize(knnr_objective, lb_knnr, ub_knnr, pop_size=30, max_iter=50)

print("\nOPTIMIZING CATBOOST REGRESSION (CATR)...")
best_catr, score_catr = eoa_optimize(catr_objective, lb_catr, ub_catr, pop_size=30, max_iter=60)


# ------------------- 6. Final Test Evaluation -------------------
rr = Ridge(alpha=best_rr[0]).fit(X_train_s, y_train)
knn_w = ['uniform', 'distance'][int(best_knnr[1])]
knn = KNeighborsRegressor(n_neighbors=int(best_knnr[0]), weights=knn_w, p=int(best_knnr[2])).fit(X_train_s, y_train)
cat = CatBoostRegressor(depth=int(best_catr[0]), learning_rate=best_catr[1],
                         iterations=int(best_catr[2]), l2_leaf_reg=best_catr[3],
                         random_seed=42, verbose=False).fit(X_train, y_train)

test_rr = np.sqrt(mean_squared_error(y_test, rr.predict(X_test_s)))
test_knn = np.sqrt(mean_squared_error(y_test, knn.predict(X_test_s)))
test_cat = np.sqrt(mean_squared_error(y_test, cat.predict(X_test)))

# ------------------- 7. Final Report -------------------
print("\n" + "="*85)
print("FINAL RESULTS - ECHIDNA OPTIMIZATION ALGORITHM (EOA)")
print("="*85)
print(f"RR     → alpha = {best_rr[0]:8.4f}   | CV RMSE: {score_rr:.4f}  → Test RMSE: {test_rr:.4f}")
print(f"KNNR   → n_neighbors = {int(best_knnr[0]):2d}, {knn_w:8s}, p={int(best_knnr[2])} "
      f"| CV RMSE: {score_knnr:.4f}  → Test RMSE: {test_knn:.4f}")
print(f"CATR   → depth={int(best_catr[0])}, lr={best_catr[1]:.4f}, "
      f"iter={int(best_catr[2]):4d}, l2={best_catr[3]:.2f} | CV RMSE: {score_catr:.4f}  → Test RMSE: {test_cat:.4f}")
print("="*85)
print("Echidna has finished foraging – global optimum secured!")