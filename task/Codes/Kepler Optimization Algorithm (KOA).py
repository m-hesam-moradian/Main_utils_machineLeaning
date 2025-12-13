import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.special import gamma
from math import pi, sin, exp

# Load sample dataset for demonstration (UCI Adult for classification)
print("Loading UCI Adult dataset...")
data = fetch_openml(name='adult', version=2, as_frame=True)
X, y = data.data, data.target
y = LabelEncoder().fit_transform(y)  # '>50K' → 1

# Preprocess: one-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

# Scale features (important for LR, KNNC, GPC)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Kepler Optimization Algorithm (KOA) implementation
def koa_optimize(objective_func, lb, ub, N=30, T=100, mu0=0.5, gamma=0.1):
    """
    Kepler Optimization Algorithm for minimizing an objective function.
    Inspired by Kepler's laws of planetary motion, with gravitational force, orbital velocity, and position updates.
    """
    D = len(lb)
    # Initialize population (planets)
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Initialize eccentricity and orbital period
    e = np.random.rand(N)  # eccentricity
    T_period = np.abs(np.random.randn(N))  # orbital period

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        # Adaptive mu
        mu = mu0 * exp(-gamma * t / T)

        # Find min f (f_s), worst
        f_s = np.min(fitness)
        worst = np.max(fitness)
        sum_diff = np.sum(fitness - worst)

        # Masses
        M_s = (f_s - worst) / sum_diff if sum_diff != 0 else 1.0 / N
        m = np.random.rand(N) * (fitness - worst) / sum_diff if sum_diff != 0 else np.ones(N) / N
        bar_M_s = M_s / max(M_s, 1e-10)
        bar_m = m / np.max(m)

        # For each planet
        for i in range(N):
            # Distance r
            r = np.linalg.norm(population[i] - best_params)
            bar_r = r / (np.max(np.linalg.norm(population - best_params, axis=1)) - np.min(np.linalg.norm(population - best_params, axis=1)) + 1e-10)

            # Gravitational force F
            r1 = np.random.rand()
            F = e[i] * mu * bar_M_s * bar_m[i] * (bar_r ** 2) / (bar_r + 1e-10 + r1)

            # Semi-major axis a
            a = (T_period[i] ** 2 * mu * (bar_M_s + bar_m[i]) / (4 * pi ** 2)) ** (1/3)

            # Orbital velocity v
            v = np.sqrt(mu * (2 / r - 1 / a + 1e-10))

            # Random X_a, X_b from population
            a_idx = np.random.randint(N)
            b_idx = np.random.randint(N)
            X_a = population[a_idx]
            X_b = population[b_idx]

            # r3, r4, r5, sigma
            r3 = np.random.rand()
            r4 = np.random.rand()
            r5 = np.random.rand()
            sigma = 1 if r4 <= 0.5 else -1

            # Velocity update based on bar r
            if bar_r <= 0.5:
                # Exploitation
                M = r3 * (1 - r4) + r4
                kappa = v
                delta = 0 if r5 <= r3 else 1 * M * kappa  # simplified U
                delta_pp = 1 - delta
                v_vec = delta * (2 * r4 * (population[i] - X_b) + delta_pp * (X_a - X_b)) + (1 - bar_r) * sigma * (1 if r5 <= r4 else 0) * r5 * (ub - lb)
            else:
                # Exploration
                v_vec = r4 * v * (X_a - population[i]) + (1 - bar_r) * sigma * (1 if r3 <= r4 else 0) * r5 * r3 * (ub - lb)

            # Position update
            r = np.random.rand()
            U = 0 if r5 <= r3 else 1  # simplified
            new_pos = population[i] + sigma * v_vec + U * (F + r) * (best_params - population[i])

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)

            # Update if better
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t}: Best params = {best_params}, Best error = {best_score:.4f}")

        # Elitism already in update

    return best_params, best_score


# Objective function for Logistic Regression (LR)
def objective_lr(params):
    """
    Objective for LogisticRegression.
    Params: [C, penalty_index (0: l1, 1: l2)]
    """
    C = params[0]
    penalty_idx = int(params[1])
    penalty = ['l1', 'l2'][penalty_idx]
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'

    model = LogisticRegression(
        C=C, penalty=penalty, solver=solver, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score


# Objective function for K-Nearest Neighbors Classification (KNNC)
def objective_knnc(params):
    """
    Objective for KNeighborsClassifier.
    Params: [n_neighbors, weights_index (0: uniform, 1: distance), p]
    """
    n_neighbors = int(params[0])
    weights_idx = int(params[1])
    p = int(params[2])

    weights = ['uniform', 'distance'][weights_idx]

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, p=p
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score


# Objective function for Gaussian Process Classification (GPC)
def objective_gpc(params):
    """
    Objective for GaussianProcessClassifier.
    Params: [length_scale, n_restarts_optimizer]
    """
    length_scale = params[0]
    n_restarts = int(params[1])

    kernel = RBF(length_scale=length_scale)

    model = GaussianProcessClassifier(
        kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score


# Hyperparameter bounds for LR
lb_lr = np.array([0.01, 0])
ub_lr = np.array([10, 1])

# Hyperparameter bounds for KNNC
lb_knnc = np.array([3, 0, 1])
ub_knnc = np.array([30, 1, 2])

# Hyperparameter bounds for GPC
lb_gpc = np.array([0.1, 0])
ub_gpc = np.array([10, 10])

# Optimize LR using KOA
print("Optimizing LR with KOA...")
best_params_lr, best_error_lr = koa_optimize(
    objective_lr, lb_lr, ub_lr, N=20, T=50
)
print(
    f"Best LR params: C={best_params_lr[0]:.4f}, penalty={['l1', 'l2'][int(best_params_lr[1])]}"
)
print(f"Best CV error: {best_error_lr:.4f}")

# Train final LR model and evaluate on test set
penalty = ['l1', 'l2'][int(best_params_lr[1])]
solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
lr_final = LogisticRegression(
    C=best_params_lr[0],
    penalty=penalty,
    solver=solver,
    random_state=42
)
lr_final.fit(X_train, y_train)
y_pred_lr = lr_final.predict(X_test)
test_acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Test accuracy for LR: {test_acc_lr:.4f}\n")

# Optimize KNNC using KOA
print("Optimizing KNNC with KOA...")
best_params_knnc, best_error_knnc = koa_optimize(
    objective_knnc, lb_knnc, ub_knnc, N=20, T=50
)
print(
    f"Best KNNC params: n_neighbors={int(best_params_knnc[0])}, weights={['uniform', 'distance'][int(best_params_knnc[1])]}, p={int(best_params_knnc[2])}"
)
print(f"Best CV error: {best_error_knnc:.4f}")

# Train final KNNC model and evaluate on test set
weights = ['uniform', 'distance'][int(best_params_knnc[1])]
knnc_final = KNeighborsClassifier(
    n_neighbors=int(best_params_knnc[0]),
    weights=weights,
    p=int(best_params_knnc[2])
)
knnc_final.fit(X_train, y_train)
y_pred_knnc = knnc_final.predict(X_test)
test_acc_knnc = accuracy_score(y_test, y_pred_knnc)
print(f"Test accuracy for KNNC: {test_acc_knnc:.4f}\n")

# Optimize GPC using KOA
print("Optimizing GPC with KOA...")
best_params_gpc, best_error_gpc = koa_optimize(
    objective_gpc, lb_gpc, ub_gpc, N=20, T=50
)
print(
    f"Best GPC params: length_scale={best_params_gpc[0]:.4f}, n_restarts_optimizer={int(best_params_gpc[1])}"
)
print(f"Best CV error: {best_error_gpc:.4f}")

# Train final GPC model and evaluate on test set
kernel = RBF(length_scale=best_params_gpc[0])
gpc_final = GaussianProcessClassifier(
    kernel=kernel,
    n_restarts_optimizer=int(best_params_gpc[1]),
    random_state=42
)
gpc_final.fit(X_train, y_train)
y_pred_gpc = gpc_final.predict(X_test)
test_acc_gpc = accuracy_score(y_test, y_pred_gpc)
print(f"Test accuracy for GPC: {test_acc_gpc:.4f}")