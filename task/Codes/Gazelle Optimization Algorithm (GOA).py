# --------------------------------------------------------------
#  Gazelle Optimization Algorithm (GOA) + LGBC + ETC
# --------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# No scaling needed for tree-based models (LGBC, ETC)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------- 2. Gazelle Optimization Algorithm (GOA) -------------------
def goa_optimize(obj_func, lb, ub, pop_size=30, max_iter=50, verbose=True):
    """
    Gazelle Optimization Algorithm (GOA)
    -------------------------------------------------
    Inspired by gazelle herd behavior:
    • Swift Movement     → Lévy flight jumps (exploration)
    • Group Coordination → Move toward herd center & leader
    • Predator Avoidance → Random escape jumps
    • Herd Memory        → Elite replacement
    -------------------------------------------------
    """
    dim = len(lb)
    # Initialize gazelle herd
    herd = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in herd])

    gbest_idx = np.argmin(fitness)
    gbest = herd[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best Error = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for it in range(1, max_iter + 1):
        alpha = 1.0 - it / max_iter  # Exploration → exploitation
        herd_center = np.mean(herd, axis=0)

        new_herd = herd.copy()

        for i in range(pop_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            # ---- Phase 1: Swift Movement (Lévy flight) ----
            if np.random.rand() < alpha:
                beta = 1.5
                sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                         (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
                u = np.random.randn(dim) * sigma
                v = np.random.randn(dim)
                step = u / (np.abs(v) ** (1 / beta))
                candidate = herd[i] + 0.01 * step * (ub - lb) * alpha
            else:
                # ---- Phase 2: Group Coordination ----
                to_leader = r1 * (gbest - herd[i])
                to_center = r2 * (herd_center - herd[i])
                candidate = herd[i] + alpha * (to_leader + 0.5 * to_center)

            # ---- Phase 3: Predator Avoidance (random escape) ----
            if np.random.rand() < 0.1:
                escape = (ub - lb) * (np.random.rand(dim) - 0.5) * 0.3
                candidate += escape

            # Clip to bounds
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            new_fit = obj_func(candidate)

            # Greedy selection
            if new_fit < fitness[i]:
                new_herd[i] = candidate
                fitness[i] = new_fit
                if new_fit < gbest_fit:
                    gbest = candidate.copy()
                    gbest_fit = new_fit
                    if verbose:
                        print(f"Iter {it:02d} | Best Error = {gbest_fit:.5f} | "
                              f"Params = {np.round(gbest, 4)}")

        herd = new_herd

        # ---- Phase 4: Herd Memory (replace worst) ----
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.12:
            herd[worst_idx] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst_idx] = obj_func(herd[worst_idx])

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


# ------------------- 5. Run GOA -------------------
print("\n" + "="*70)
print("OPTIMIZING LIGHTGBM CLASSIFIER (LGBC) WITH GOA")
print("="*70)
best_lgbc, err_lgbc = goa_optimize(lgbc_objective, lb_lgbc, ub_lgbc, pop_size=25, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING EXTRA TREES CLASSIFIER (ETC) WITH GOA")
print("="*70)
best_etc, err_etc = goa_optimize(etc_objective, lb_etc, ub_etc, pop_size=25, max_iter=40)


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
print("FINAL RESULTS (Gazelle Optimization Algorithm - GOA)")
print("="*70)
print(f"LGBC   → leaves={int(best_lgbc[0])}, lr={best_lgbc[1]:.4f}, "
      f"n_est={int(best_lgbc[2])}, depth={int(best_lgbc[3])} | "
      f"CV error={err_lgbc:.4f} | Test Acc={acc_lgbc:.4f}")
print(f"ETC    → n_est={int(best_etc[0])}, max_depth={int(best_etc[1])}, "
      f"min_split={int(best_etc[2])}, min_leaf={int(best_etc[3])} | "
      f"CV error={err_etc:.4f} | Test Acc={acc_etc:.4f}")
print("="*70)