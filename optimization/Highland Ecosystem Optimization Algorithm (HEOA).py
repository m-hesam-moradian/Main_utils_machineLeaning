# --------------------------------------------------------------
#  Highland Ecosystem Optimization Algorithm (HEOA) - 2024
#  + SVC + Logistic Regression + Gaussian Process Classification
# --------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import warnings
warnings.filterwarnings("ignore")

# ------------------- 1. Load Adult Dataset -------------------
print("Loading UCI Adult dataset...")
data = fetch_openml(name='adult', version=2, as_frame=True)
X, y = data.data, data.target
y = LabelEncoder().fit_transform(y)  # >50K → 1

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Scale features (critical for SVC and GPC)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ------------------- 2. Highland Ecosystem Optimization Algorithm (HEOA) -------------------
def heoa_optimize(obj_func, lb, ub, pop_size=35, max_iter=60, verbose=True):
    """
    Highland Ecosystem Optimization Algorithm (HEOA) - 2024
    Inspired by high-altitude mountain ecosystem dynamics
    """
    dim = len(lb)
    herd = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in herd])

    gbest_idx = np.argmin(fitness)
    gbest = herd[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    if verbose:
        print(f"Iter 00 | Best Error = {gbest_fit:.5f} | Params = {np.round(gbest, 4)}")

    for t in range(1, max_iter + 1):
        altitude = 1.0 - (t / max_iter)**1.5  # Non-linear descent from highlands

        # Apex predators (top 15% best individuals)
        predator_num = max(2, int(0.15 * pop_size))
        predator_idx = np.argsort(fitness)[:predator_num]
        apex_center = np.mean(herd[predator_idx], axis=0)

        new_herd = np.zeros_like(herd)

        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()

            # ---- Predator Pursuit (move toward apex predators) ----
            if r1 < 0.65:
                direction = apex_center - herd[i]
                candidate = herd[i] + altitude * r2 * direction
            else:
                # ---- Prey Evasion (Lévy flight escape) ----
                beta = 1.5
                sigma = (np.gamma(1+beta)*np.sin(np.pi*beta/2) /
                         (np.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                levy_step = 0.02 * np.random.randn(dim) * sigma / (np.abs(np.random.randn(dim))**(1/beta))
                candidate = herd[i] + levy_step * (ub - lb) * altitude

            # ---- Seasonal Spiral Migration (golden angle spiral) ----
            theta = 2 * np.pi * t * 0.618034  # Golden ratio conjugate
            radius = altitude * np.random.rand()
            spiral = radius * np.array([np.cos(theta), np.sin(theta)])
            if dim > 2:
                spiral = np.tile(spiral, (dim + 1)//2)[:dim]
            candidate += spiral * (ub - lb) * 0.05

            # ---- Alpine Adaptation (refine toward global best) ----
            if np.random.rand() < 0.3:
                candidate += 0.08 * altitude * (gbest - candidate)

            # ---- Resource Scarcity Pressure (push away from median) ----
            if fitness[i] > np.median(fitness):
                candidate += 0.12 * (ub - lb) * (np.random.rand(dim) - 0.5)

            candidate = np.clip(candidate, lb, ub)
            new_fit = obj_func(candidate)

            if new_fit < fitness[i]:
                new_herd[i] = candidate
                fitness[i] = new_fit
                if new_fit < gbest_fit:
                    gbest = candidate.copy()
                    gbest_fit = new_fit
                    if verbose:
                        print(f"Iter {t:02d} | Best Error = {1-gbest_fit:.5f} Acc | Params = {np.round(gbest, 4)}")
            else:
                new_herd[i] = herd[i]

        herd = new_herd

        # ---- New Migrants from Lower Valleys (immigration) ----
        if np.random.rand() < 0.1:
            worst = np.argmax(fitness)
            herd[worst] = lb + np.random.rand(dim) * (ub - lb)
            fitness[worst] = obj_func(herd[worst])

    return gbest, gbest_fit


# ------------------- 3. Objective Functions (return 1 - accuracy) -------------------
def svc_objective(p):
    C, gamma, k_idx = p[0], p[1], int(p[2])
    kernel = ['rbf', 'poly', 'sigmoid'][k_idx]
    model = SVC(C=C, gamma=gamma, kernel=kernel, max_iter=1000, random_state=42)
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=4).mean()
    return 1.0 - acc

def lr_objective(p):
    C, pen_idx = p[0], int(p[1])
    penalty = ['l1', 'l2'][pen_idx]
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=4).mean()
    return 1.0 - acc

def gpc_objective(p):
    length_scale, n_restarts = p[0], int(p[1])
    kernel = 1.0 * RBF(length_scale=length_scale)
    model = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42)
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=4).mean()
    return 1.0 - acc


# ------------------- 4. Bounds -------------------
lb_svc = np.array([0.1, 0.001, 0])
ub_svc = np.array([100.0, 10.0, 2])

lb_lr = np.array([0.01, 0])
ub_lr = np.array([100.0, 1])

lb_gpc = np.array([0.1, 0])
ub_gpc = np.array([10.0, 10])


# ------------------- 5. Run HEOA -------------------
print("\n" + "="*90)
print("HIGHLAND ECOSYSTEM OPTIMIZATION ALGORITHM (HEOA) - 2024")
print("="*90)

print("\nOptimizing Support Vector Classification (SVC)...")
best_svc, err_svc = heoa_optimize(svc_objective, lb_svc, ub_svc, pop_size=35, max_iter=60)

print("\nOptimizing Logistic Regression (LR)...")
best_lr, err_lr = heoa_optimize(lr_objective, lb_lr, ub_lr, pop_size=35, max_iter=50)

print("\nOptimizing Gaussian Process Classification (GPC)...")
best_gpc, err_gpc = heoa_optimize(gpc_objective, lb_gpc, ub_gpc, pop_size=35, max_iter=70)


# ------------------- 6. Final Evaluation -------------------
# SVC
kernel_name = ['rbf', 'poly', 'sigmoid'][int(best_svc[2])]
svc_final = SVC(C=best_svc[0], gamma=best_svc[1], kernel=kernel_name, max_iter=1000)
svc_final.fit(X_train, y_train)
acc_svc = accuracy_score(y_test, svc_final.predict(X_test))

# LR
pen_name = ['l1', 'l2'][int(best_lr[1])]
solver = 'liblinear' if pen_name == 'l1' else 'lbfgs'
lr_final = LogisticRegression(C=best_lr[0], penalty=pen_name, solver=solver, max_iter=1000)
lr_final.fit(X_train, y_train)
acc_lr = accuracy_score(y_test, lr_final.predict(X_test))

# GPC
gpc_kernel = 1.0 * RBF(length_scale=best_gpc[0])
gpc_final = GaussianProcessClassifier(kernel=gpc_kernel, n_restarts_optimizer=int(best_gpc[1]), random_state=42)
gpc_final.fit(X_train, y_train)
acc_gpc = accuracy_score(y_test, gpc_final.predict(X_test))


# ------------------- 7. Final Report -------------------
print("\n" + "="*90)
print("FINAL RESULTS - HIGHLAND ECOSYSTEM OPTIMIZATION ALGORITHM (HEOA)")
print("="*90)
print(f"SVC  | C={best_svc[0]:7.3f}, gamma={best_svc[1]:.4f}, kernel={kernel_name:8s} | "
      f"CV Error={err_svc:.4f} → Test Acc = {acc_svc:.4f}")
print(f"LR   | C={best_lr[0]:7.3f}, penalty={pen_name:2s}                           | "
      f"CV Error={err_lr:.4f} → Test Acc = {acc_lr:.4f}")
print(f"GPC  | RBF lengthscale={best_gpc[0]:6.3f}, restarts={int(best_gpc[1]):2d}       | "
      f"CV Error={err_gpc:.4f} → Test Acc = {acc_gpc:.4f}")
print("="*90)
print("HEOA has reached the summit — global optimum conquered from the highlands!")