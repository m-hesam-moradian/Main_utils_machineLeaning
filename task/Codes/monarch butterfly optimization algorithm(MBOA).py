# --------------------------------------------------------------
#  Mother Optimization Algorithm (MOA) + XGBC + SVC + DTC
# --------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ------------------- 1. Load & preprocess Adult dataset -------------------
print("Loading Adult dataset...")
data = fetch_openml(name='adult', version=2, as_frame=True)
X, y = data.data, data.target
y = LabelEncoder().fit_transform(y)  # '>50K' → 1

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Scale for SVC
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------- 2. Mother Optimization Algorithm (MOA) -------------------
def moa_optimize(obj_func, lb, ub, pop_size=25, max_iter=50, verbose=True):
    """
    Mother Optimization Algorithm (MOA)
    -------------------------------------------------
    Inspired by a mother's nurturing behavior:
    • Teaching          → Move toward best child (gbest)
    • Caring            → Help weaker children
    • Protecting        → Emergency Lévy jumps
    • Family Gathering  → Elite replacement
    • Love Factor       → Adaptive balance
    -------------------------------------------------
    """
    dim = len(lb)
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])

    gbest_idx = np.argmin(fitness)
    gbest = pop[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best Error = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for it in range(1, max_iter + 1):
        love = 1.0 - it / max_iter  # Love decreases → more independence

        for i in range(pop_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            # ---- Phase 1: Teaching (learn from best child) ----
            if np.random.rand() < 0.7:
                step = love * r1 * (gbest - pop[i])
            else:
                # ---- Phase 2: Caring (help weaker sibling) ----
                weak_idx = np.random.choice(np.argsort(fitness)[-3:])  # bottom 3
                step = love * r2 * (pop[weak_idx] - pop[i])

            # ---- Phase 3: Gentle Guidance (local love) ----
            guidance = (ub - lb) * np.random.randn(dim) * 0.02 * (1 - love)
            candidate = pop[i] + step + guidance

            # ---- Phase 4: Protection (emergency Lévy jump) ----
            if fitness[i] > np.mean(fitness) and np.random.rand() < 0.15:
                levy = 0.01 * np.random.pareto(1.5, dim) * np.sign(np.random.randn(dim))
                candidate += levy * (ub - lb) * love

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
                        print(f"Iter {it:02d} | Best Error = {gbest_fit:.5f} | "
                              f"Params = {np.round(gbest, 4)}")

        # ---- Phase 5: Family Gathering (replace worst) ----
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.1:
            pop[worst_idx] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst_idx] = obj_func(pop[worst_idx])

    return gbest, gbest_fit


# ------------------- 3. Objective Functions -------------------
def xgbc_objective(params):
    max_depth = int(params[0])
    lr = params[1]
    n_est = int(params[2])
    subsample = params[3]
    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=lr,
        n_estimators=n_est,
        subsample=subsample,
        random_state=42,
        n_jobs=4,
        verbosity=0
    )
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return 1.0 - acc

def svc_objective(params):
    C = params[0]
    gamma = params[1]
    kernel_idx = int(params[2])
    kernel = ['rbf', 'poly', 'sigmoid'][kernel_idx]
    model = SVC(C=C, gamma=gamma, kernel=kernel, max_iter=1000)
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return 1.0 - acc

def dtc_objective(params):
    max_depth = int(params[0])
    min_split = int(params[1])
    min_leaf = int(params[2])
    model = DecisionTreeClassifier(
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        random_state=42
    )
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return 1.0 - acc


# ------------------- 4. Search Bounds -------------------
lb_xgbc = np.array([3, 0.01, 50, 0.5])
ub_xgbc = np.array([15, 0.3, 300, 1.0])

lb_svc = np.array([0.1, 0.001, 0])
ub_svc = np.array([100.0, 10.0, 2])

lb_dtc = np.array([1, 2, 1])
ub_dtc = np.array([30, 20, 10])


# ------------------- 5. Run MOA -------------------
print("\n" + "="*70)
print("OPTIMIZING XGBOOST CLASSIFIER (XGBC) WITH MOA")
print("="*70)
best_xgbc, err_xgbc = moa_optimize(xgbc_objective, lb_xgbc, ub_xgbc, pop_size=20, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING SUPPORT VECTOR CLASSIFIER (SVC) WITH MOA")
print("="*70)
best_svc, err_svc = moa_optimize(svc_objective, lb_svc, ub_svc, pop_size=20, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING DECISION TREE CLASSIFIER (DTC) WITH MOA")
print("="*70)
best_dtc, err_dtc = moa_optimize(dtc_objective, lb_dtc, ub_dtc, pop_size=20, max_iter=40)


# ------------------- 6. Final Evaluation -------------------
# XGBC
xgbc_final = XGBClassifier(
    max_depth=int(best_xgbc[0]),
    learning_rate=best_xgbc[1],
    n_estimators=int(best_xgbc[2]),
    subsample=best_xgbc[3],
    random_state=42,
    n_jobs=4,
    verbosity=0
)
xgbc_final.fit(X_train, y_train)
acc_xgbc = accuracy_score(y_test, xgbc_final.predict(X_test))

# SVC
kernel = ['rbf', 'poly', 'sigmoid'][int(best_svc[2])]
svc_final = SVC(C=best_svc[0], gamma=best_svc[1], kernel=kernel, max_iter=1000)
svc_final.fit(X_train, y_train)
acc_svc = accuracy_score(y_test, svc_final.predict(X_test))

# DTC
dtc_final = DecisionTreeClassifier(
    max_depth=int(best_dtc[0]) if best_dtc[0] > 0 else None,
    min_samples_split=int(best_dtc[1]),
    min_samples_leaf=int(best_dtc[2]),
    random_state=42
)
dtc_final.fit(X_train, y_train)
acc_dtc = accuracy_score(y_test, dtc_final.predict(X_test))


# ------------------- 7. Final Report -------------------
print("\n" + "="*70)
print("FINAL RESULTS (Mother Optimization Algorithm - MOA)")
print("="*70)
print(f"XGBC   → depth={int(best_xgbc[0])}, lr={best_xgbc[1]:.4f}, "
      f"n_est={int(best_xgbc[2])}, subsample={best_xgbc[3]:.3f} | "
      f"CV error={err_xgbc:.4f} | Test Acc={acc_xgbc:.4f}")
print(f"SVC    → C={best_svc[0]:.4f}, gamma={best_svc[1]:.4f}, kernel={kernel} | "
      f"CV error={err_svc:.4f} | Test Acc={acc_svc:.4f}")
print(f"DTC    → depth={int(best_dtc[0])}, min_split={int(best_dtc[1])}, "
      f"min_leaf={int(best_dtc[2])} | CV error={err_dtc:.4f} | Test Acc={acc_dtc:.4f}")
print("="*70)