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
import math

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


# Red Panda Optimization Algorithm (RPOA) implementation
def rpoa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Red Panda Optimization Algorithm for minimizing an objective function.
    Based on red panda foraging (exploration) and climbing/resting (exploitation) behaviors.
    """
    D = len(lb)
    # Initialize population
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        for i in range(N):
            # Phase 1: Exploration (Foraging)
            # Identify better solutions as food sources
            better_indices = [k for k in range(N) if fitness[k] < fitness[i] and k != i]
            food_sources = [population[k] for k in better_indices]
            if len(food_sources) > 0:
                food_i = food_sources[np.random.randint(len(food_sources))]
            else:
                food_i = best_params

            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            new_pos = population[i] + r1 * (food_i - population[i]) + r2 * (best_params - population[i])

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate and update if better
            new_fit = objective_func(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t}: Best params = {best_params}, Best error = {best_score:.4f}")

            # Phase 2: Exploitation (Climbing and Resting)
            r = np.random.rand(D)
            decay = math.exp(-t / T)
            new_pos = population[i] + r * (best_params - population[i]) * decay

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate and update if better
            new_fit = objective_func(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t}: Best params = {best_params}, Best error = {best_score:.4f}")

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

# Optimize LR using RPOA
print("Optimizing LR with RPOA...")
best_params_lr, best_error_lr = rpoa_optimize(
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

# Optimize KNNC using RPOA
print("Optimizing KNNC with RPOA...")
best_params_knnc, best_error_knnc = rpoa_optimize(
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

# Optimize GPC using RPOA
print("Optimizing GPC with RPOA...")
best_params_gpc, best_error_gpc = rpoa_optimize(
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