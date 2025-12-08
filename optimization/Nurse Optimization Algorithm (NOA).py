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

# Load sample dataset for demonstration (UCI Adult for classification)
print("Loading UCI Adult dataset...")
data = fetch_openml(name='adult', version=2, as_frame=True)
X, y = data.data, data.target
y = LabelEncoder().fit_transform(y)  # '>50K' → 1

# Preprocess: one-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

# Scale features (important for SVC, LR, GPC)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Nurse Optimization Algorithm (NOA) implementation
def noa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Nurse Optimization Algorithm for minimizing an objective function.
    Inspired by nursing behaviors: patient monitoring (exploration), team rounds (coordination), adaptive treatment (exploitation), emergency response (diversification), and shift change (elite replacement).
    """
    D = len(lb)
    population = lb + np.random.rand(N, D) * (ub - lb)

    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        # Adaptive care level: exploration to exploitation
        care_level = 1 - t / T

        for i in range(N):
            # Phase 1: Patient Monitoring (Exploration: large steps with Lévy-like simulation)
            if np.random.rand() < care_level:
                step_size = (ub - lb) * np.random.pareto(1.5, D) / (t ** 0.5)
                direction = np.sign(np.random.randn(D))
                new_pos = population[i] + direction * np.random.rand(D) * step_size
            else:
                # Phase 2: Team Rounds (Exploitation: converge to best)
                r = np.random.rand(D)
                convergence_factor = (1 - care_level) * r
                new_pos = population[i] + convergence_factor * (best_params - population[i])

            # Phase 3: Adaptive Treatment (Local perturbation)
            perturbation = (ub - lb) * np.random.randn(D) * (1 / (t + 1))
            new_pos += perturbation

            # Phase 4: Emergency Response (Diversification if stuck)
            if np.random.rand() < 0.1 * care_level:
                emergency_strength = (ub - lb) * np.random.uniform(0.05, 0.2, D)
                new_pos += emergency_strength * np.random.randn(D)

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

        # Phase 5: Shift Change (Elite replacement to avoid stagnation)
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.05:
            population[worst_idx] = lb + np.random.rand(D) * (ub - lb)
            fitness[worst_idx] = objective_func(population[worst_idx])

    return best_params, best_score


# Objective function for SVC
def objective_svc(params):
    """
    Objective for SVC.
    Params: [C, gamma, kernel_index (0: rbf, 1: poly, 2: sigmoid)]
    """
    C = params[0]
    gamma = params[1]
    kernel_idx = int(params[2])
    kernel = ['rbf', 'poly', 'sigmoid'][kernel_idx]

    model = SVC(
        C=C, gamma=gamma, kernel=kernel, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score  # Minimize 1 - accuracy


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


# Objective function for Gaussian Process Classifier (GPC)
def objective_gpc(params):
    """
    Objective for GaussianProcessClassifier.
    Params: [length_scale for RBF, n_restarts_optimizer]
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


# Hyperparameter bounds for SVC
lb_svc = np.array([0.1, 0.001, 0])
ub_svc = np.array([100, 1, 2])

# Hyperparameter bounds for LR
lb_lr = np.array([0.1, 0])
ub_lr = np.array([100, 1])

# Hyperparameter bounds for GPC
lb_gpc = np.array([0.1, 0])
ub_gpc = np.array([10, 10])

# Optimize SVC using NOA
print("Optimizing SVC with NOA...")
best_params_svc, best_error_svc = noa_optimize(
    objective_svc, lb_svc, ub_svc, N=20, T=50
)
print(
    f"Best SVC params: C={best_params_svc[0]:.4f}, gamma={best_params_svc[1]:.4f}, kernel={['rbf', 'poly', 'sigmoid'][int(best_params_svc[2])]}"
)
print(f"Best CV error: {best_error_svc:.4f}")

# Train final SVC model and evaluate on test set
kernel = ['rbf', 'poly', 'sigmoid'][int(best_params_svc[2])]
svc_final = SVC(
    C=best_params_svc[0],
    gamma=best_params_svc[1],
    kernel=kernel,
    random_state=42
)
svc_final.fit(X_train, y_train)
y_pred_svc = svc_final.predict(X_test)
test_acc_svc = accuracy_score(y_test, y_pred_svc)
print(f"Test accuracy for SVC: {test_acc_svc:.4f}\n")

# Optimize LR using NOA
print("Optimizing LR with NOA...")
best_params_lr, best_error_lr = noa_optimize(
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

# Optimize GPC using NOA
print("Optimizing GPC with NOA...")
best_params_gpc, best_error_gpc = noa_optimize(
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
