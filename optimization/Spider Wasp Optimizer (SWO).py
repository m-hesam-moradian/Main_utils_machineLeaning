# --------------------------------------------------------------
#  Spider Wasp Optimizer (SWO) + LR + ETC + ADAC
# --------------------------------------------------------------
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from scipy.stats import levy
import warnings
warnings.filterwarnings("ignore")

# ------------------- 1. Load & preprocess Adult dataset -------------------
print("Loading Adult dataset...")
data = fetch_openml(name='adult', version=2, as_frame=True)
X, y = data.data, data.target
y = LabelEncoder().fit_transform(y)  # '>50K' → 1, '<=50K' → 0

# Handle categorical features
X = pd.get_dummies(X, drop_first=True)

# Scale features (important for LR)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ------------------- 2. Spider Wasp Optimizer (SWO) -------------------
def swo_optimize(obj_func, lb, ub, pop_size=25, max_iter=50, verbose=True):
    """
    Spider Wasp Optimizer (SWO)
    -------------------------------------------------
    • Hunting   → Lévy-flight long jumps (exploration)
    • Paralyzing→ Move toward global best (exploitation)
    • Nest-building → Tiny local refinement
    • Elite replacement → Avoid stagnation
    -------------------------------------------------
    Returns: best_params, best_score (negative accuracy)
    """
    dim = len(lb)
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])

    gbest_idx = np.argmin(fitness)
    gbest = pop[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best fitness = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for it in range(1, max_iter + 1):
        alpha = 1.0 - it / max_iter  # exploration → exploitation

        for i in range(pop_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            # ---- Phase 1: Hunting (Lévy jumps) ----
            if np.random.rand() < alpha:
                step = levy.rvs(size=dim, loc=0, scale=1) * (ub - lb) * alpha
                candidate = pop[i] + step
            else:
                # ---- Phase 2: Paralyzing strike ----
                candidate = pop[i] + r1 * (gbest - pop[i]) * (1 - alpha)

            # ---- Phase 3: Nest-building (local refinement) ----
            local_noise = (ub - lb) * np.random.randn(dim) * 0.02 * (1 - alpha)
            candidate += local_noise

            # Clip to bounds
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
                        print(f"Iter {it:02d} | Best fitness = {gbest_fit:.5f} | "
                              f"Params = {np.round(gbest, 4)}")

        # ---- Elite replacement (avoid stagnation) ----
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.1:
            pop[worst_idx] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst_idx] = obj_func(pop[worst_idx])

    return gbest, gbest_fit


# ------------------- 3. Objective Functions -------------------
def lr_objective(params):
    C = params[0]
    penalty_idx = int(params[1])  # 0: l1, 1: l2
    penalty = ['l1', 'l2'][penalty_idx]
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return 1.0 - acc

def etc_objective(params):
    n_est = int(params[0])
    max_d = int(params[1])
    min_split = int(params[2])
    model = ExtraTreesClassifier(
        n_estimators=n_est,
        max_depth=max_d if max_d > 0 else None,
        min_samples_split=min_split,
        random_state=42,
        n_jobs=4
    )
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return 1.0 - acc

def adac_objective(params):
    n_est = int(params[0])
    lr = params[1]
    model = AdaBoostClassifier(
        n_estimators=n_est,
        learning_rate=lr,
        random_state=42
    )
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return 1.0 - acc


# ------------------- 4. Search Bounds -------------------
lb_lr = np.array([0.01, 0])        # C, penalty_idx
ub_lr = np.array([100.0, 1])

lb_etc = np.array([50, 5, 2])      # n_estimators, max_depth, min_samples_split
ub_etc = np.array([300, 50, 20])

lb_adac = np.array([50, 0.01])     # n_estimators, learning_rate
ub_adac = np.array([300, 2.0])


# ------------------- 5. Run SWO -------------------
print("\n" + "="*70)
print("OPTIMIZING LOGISTIC REGRESSION (LR) WITH SWO")
print("="*70)
best_lr, err_lr = swo_optimize(lr_objective, lb_lr, ub_lr, pop_size=20, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING EXTRA TREES CLASSIFIER (ETC) WITH SWO")
print("="*70)
best_etc, err_etc = swo_optimize(etc_objective, lb_etc, ub_etc, pop_size=20, max_iter=40)

print("\n" + "="*70)
print("OPTIMIZING ADABOOST CLASSIFIER (ADAC) WITH SWO")
print("="*70)
best_adac, err_adac = swo_optimize(adac_objective, lb_adac, ub_adac, pop_size=20, max_iter=40)


# ------------------- 6. Final Evaluation -------------------
# LR
penalty = ['l1', 'l2'][int(best_lr[1])]
solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
lr_final = LogisticRegression(C=best_lr[0], penalty=penalty, solver=solver, max_iter=1000)
lr_final.fit(X_train, y_train)
acc_lr = accuracy_score(y_test, lr_final.predict(X_test))

# ETC
etc_final = ExtraTreesClassifier(
    n_estimators=int(best_etc[0]),
    max_depth=int(best_etc[1]) if best_etc[1] > 0 else None,
    min_samples_split=int(best_etc[2]),
    random_state=42,
    n_jobs=4
)
etc_final.fit(X_train, y_train)
acc_etc = accuracy_score(y_test, etc_final.predict(X_test))

# ADAC
adac_final = AdaBoostClassifier(
    n_estimators=int(best_adac[0]),
    learning_rate=best_adac[1],
    random_state=42
)
adac_final.fit(X_train, y_train)
acc_adac = accuracy_score(y_test, adac_final.predict(X_test))


# ------------------- 7. Final Report -------------------
print("\n" + "="*70)
print("FINAL RESULTS (Spider Wasp Optimizer)")
print("="*70)
print(f"LR     → C={best_lr[0]:.4f}, penalty={penalty} | "
      f"CV error={err_lr:.4f} | Test Acc={acc_lr:.4f}")
print(f"ETC    → n_est={int(best_etc[0])}, max_depth={int(best_etc[1])}, "
      f"min_split={int(best_etc[2])} | CV error={err_etc:.4f} | Test Acc={acc_etc:.4f}")
print(f"ADAC   → n_est={int(best_adac[0])}, lr={best_adac[1]:.4f} | "
      f"CV error={err_adac:.4f} | Test Acc={acc_adac:.4f}")
print("="*70)