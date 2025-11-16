# --------------------------------------------------------------
#  Ladybug Beetle Optimization Algorithm (LBOA) + LGBC + ETC
# --------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

# ------------------- 1. Load & preprocess Adult dataset -------------------
print("Loading Adult dataset...")
data = fetch_openml(name='adult', version=2, as_frame=True)
X, y = data.data, data.target
y = LabelEncoder().fit_transform(y)  # '>50K' → 1

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------- 2. Ladybug Beetle Optimization Algorithm (LBOA) -------------------
def lboa_optimize(obj_func, lb, ub, pop_size=30, max_iter=50, verbose=True):
    """
    Ladybug Beetle Optimization Algorithm (LBOA)
    -------------------------------------------------
    Inspired by ladybug behavior:
    • Food Search        → Lévy flight (exploration)
    • Aggregation        → Move to cluster center
    • Mating             → Crossover with elite
    • Escape Response    → Random jump when threatened
    • Seasonal Migration → Elite replacement
    -------------------------------------------------
    """
    dim = len(lb)
    # Initialize ladybug population
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])

    gbest_idx = np.argmin(fitness)
    gbest = pop[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best Error = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for it in range(1, max_iter + 1):
        t = it / max_iter
        search_intensity = 1.0 - t  # High early → exploration

        # Compute cluster center (aggregation behavior)
        cluster_center = np.mean(pop, axis=0)

        new_pop = np.zeros_like(pop)

        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()

            # ---- Phase 1: Food Search (Lévy flight) ----
            if r1 < search_intensity:
                beta = 1.5
                sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                         (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
                u = np.random.randn(dim) * sigma
                v = np.random.randn(dim)
                step = u / (np.abs(v) ** (1 / beta))
                candidate = pop[i] + 0.01 * step * (ub - lb) * search_intensity
            else:
                # ---- Phase 2: Aggregation (move to cluster) ----
                candidate = pop[i] + search_intensity * r2 * (cluster_center - pop[i])

            # ---- Phase 3: Mating (crossover with elite) ----
            if np.random.rand() < 0.6:
                elite = pop[np.random.choice(np.argsort(fitness)[:max(1, pop_size//5)])]
                crossover_point = np.random.randint(0, dim)
                candidate[:crossover_point] = elite[:crossover_point]

            # ---- Phase 4: Escape Response (predator avoidance) ----
            if fitness[i] > np.mean(fitness) and np.random.rand() < 0.15:
                candidate += (ub - lb) * (np.random.rand(dim) - 0.5) * 0.3

            # Clip to bounds
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            new_fit = obj_func(candidate)

            # Greedy selection
            if new_fit < fitness[i]:
                new_pop[i] = candidate
                fitness[i] = new_fit
                if new_fit < gbest_fit:
                    gbest = candidate.copy()
                    gbest_fit = new_fit
                    if verbose:
                        print(f"Iter {it:02d} | Best Error = {gbest_fit:.5f} | "
                              f"Params = {np.round(gbest, 4)}")
            else:
                new_pop[i] = pop[i]

        pop = new_pop

        # ---- Phase 5: Seasonal Migration (elite replacement) ----
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.1:
            pop[worst_idx] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst_idx] = obj_func(pop[worst_idx])

    return gbest, gbest_fit


# ------------------- 3. Objective Functions -------------------
def lgbc_objective(params):
    num_leaves = int(params[0])
    lr = params[1]
    n_est = int(params[2])
    max_depth = int(params[3])
    model = LGBMClassifier(
        num_leaves=num_leaves,
        learning_rate=lr,
        n_estimators=n_est,
        max_depth=max_depth if max_depth > 0 else -1,
        random_state=42,
        n_jobs=4,
        verbosity=-1
    )
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return 1.0 - acc

def etc_objective(params):
    n_est = int(params[0])
    max_d = int(params[1])
    min_split = int(params[2])
    min_leaf = int(params[3])
    model = ExtraTreesClassifier(
        n_estimators=n_est,
        max_depth=max_d if max_d > 0 else None,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        random_state=42,
        n_jobs=4
    )
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return 1.0 - acc


# ------------------- 4. Search Bounds -------------------
lb_lgbc = np.array([20, 0.01, 50, 3])
ub_lgbc = np.array([300, 0.3, 500, 20])

lb_etc = np.array([50, 5, 2, 1])
ub_etc = np.array([500, 50, 20, 10])


# ------------------- 5. Run LBOA -------------------
print("\n" + "="*70)
print("OPTIMIZING LIGHTGBM CLASSIFIER (LGBC) WITH LBOA")
print("="*70)
best_lgbc, err_lgbc = lboa_optimize(lgbc_objective, lb_lgbc, ub_lgbc, pop_size=25, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING EXTRA TREES CLASSIFIER (ETC) WITH LBOA")
print("="*70)
best_etc, err_etc = lboa_optimize(etc_objective, lb_etc, ub_etc, pop_size=25, max_iter=40)


# ------------------- 6. Final Evaluation -------------------
# LGBC
lgbc_final = LGBMClassifier(
    num_leaves=int(best_lgbc[0]),
    learning_rate=best_lgbc[1],
    n_estimators=int(best_lgbc[2]),
    max_depth=int(best_lgbc[3]) if best_lgbc[3] > 0 else -1,
    random_state=42,
    n_jobs=4,
    verbosity=-1
)
lgbc_final.fit(X_train, y_train)
acc_lgbc = accuracy_score(y_test, lgbc_final.predict(X_test))

# ETC
etc_final = ExtraTreesClassifier(
    n_estimators=int(best_etc[0]),
    max_depth=int(best_etc[1]) if best_etc[1] > 0 else None,
    min_samples_split=int(best_etc[2]),
    min_samples_leaf=int(best_etc[3]),
    random_state=42,
    n_jobs=4
)
etc_final.fit(X_train, y_train)
acc_etc = accuracy_score(y_test, etc_final.predict(X_test))


# ------------------- 7. Final Report -------------------
print("\n" + "="*70)
print("FINAL RESULTS (Ladybug Beetle Optimization Algorithm - LBOA)")
print("="*70)
print(f"LGBC   → leaves={int(best_lgbc[0])}, lr={best_lgbc[1]:.4f}, "
      f"n_est={int(best_lgbc[2])}, depth={int(best_lgbc[3])} | "
      f"CV error={err_lgbc:.4f} | Test Acc={acc_lgbc:.4f}")
print(f"ETC    → n_est={int(best_etc[0])}, max_depth={int(best_etc[1])}, "
      f"min_split={int(best_etc[2])}, min_leaf={int(best_etc[3])} | "
      f"CV error={err_etc:.4f} | Test Acc={acc_etc:.4f}")
print("="*70)