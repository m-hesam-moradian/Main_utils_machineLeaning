import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

# Load sample dataset for demonstration (UCI Adult for classification)
print("Loading UCI Adult dataset...")
data = fetch_openml(name='adult', version=2, as_frame=True)
X, y = data.data, data.target
y = LabelEncoder().fit_transform(y)  # '>50K' → 1

# Preprocess: one-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Builder Optimization Algorithm (BOA) implementation
def boa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Builder Optimization Algorithm for minimizing an objective function.
    Inspired by human construction strategies: planning (exploration with random blueprints), building (exploitation toward best structure), inspection (local refinement), and reconstruction (diversification).
    """
    D = len(lb)
    # Initialize population (builders with blueprints)
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        # Adaptive build rate: exploration to exploitation
        build_rate = 1 - t / T

        for i in range(N):
            # Phase 1: Planning (Exploration: random blueprint generation)
            if np.random.rand() < build_rate:
                blueprint = lb + np.random.rand(D) * (ub - lb)
                r = np.random.rand(D)
                new_pos = population[i] + r * (blueprint - population[i])
            else:
                # Phase 2: Building (Exploitation: construct toward best)
                r = np.random.rand(D)
                new_pos = population[i] + r * (best_params - population[i])

            # Phase 3: Inspection (Local refinement with perturbation)
            perturbation = (ub - lb) * np.random.randn(D) * (1 / (t + 1))
            new_pos += perturbation

            # Phase 4: Reconstruction (Diversification if needed)
            if np.random.rand() < 0.1 * build_rate:
                recon_strength = (ub - lb) * np.random.uniform(0.1, 0.3, D)
                new_pos += recon_strength * np.random.randn(D)

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

        # Phase 5: Project Review (Elite replacement)
        worst_idx = np.argmax(fitness)
        if np.random.rand() < 0.05:
            population[worst_idx] = lb + np.random.rand(D) * (ub - lb)
            fitness[worst_idx] = objective_func(population[worst_idx])

    return best_params, best_score


# Objective function for Decision Tree Classifier (DTC)
def objective_dtc(params):
    """
    Objective for DecisionTreeClassifier.
    Params: [max_depth, min_samples_split, min_samples_leaf]
    """
    md = int(params[0])
    mss = int(params[1])
    msl = int(params[2])

    model = DecisionTreeClassifier(
        max_depth=md, min_samples_split=mss, min_samples_leaf=msl, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score


# Objective function for Extra Trees Classifier (ETC)
def objective_etc(params):
    """
    Objective for ExtraTreesClassifier.
    Params: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
    """
    ne = int(params[0])
    md = int(params[1])
    mss = int(params[2])
    msl = int(params[3])

    model = ExtraTreesClassifier(
        n_estimators=ne, max_depth=md, min_samples_split=mss, min_samples_leaf=msl, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score


# Objective function for XGBoost Classifier (XGBC)
def objective_xgbc(params):
    """
    Objective for XGBClassifier.
    Params: [max_depth, learning_rate, n_estimators, subsample]
    """
    md = int(params[0])
    lr = params[1]
    ne = int(params[2])
    ss = params[3]

    model = XGBClassifier(
        max_depth=md, learning_rate=lr, n_estimators=ne, subsample=ss, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score


# Hyperparameter bounds for DTC
lb_dtc = np.array([3, 2, 1])
ub_dtc = np.array([20, 20, 10])

# Hyperparameter bounds for ETC
lb_etc = np.array([50, 3, 2, 1])
ub_etc = np.array([300, 30, 20, 10])

# Hyperparameter bounds for XGBC
lb_xgbc = np.array([3, 0.01, 50, 0.5])
ub_xgbc = np.array([15, 0.3, 300, 1.0])

# Optimize DTC using BOA
print("Optimizing DTC with BOA...")
best_params_dtc, best_error_dtc = boa_optimize(
    objective_dtc, lb_dtc, ub_dtc, N=20, T=50
)
print(
    f"Best DTC params: max_depth={int(best_params_dtc[0])}, min_samples_split={int(best_params_dtc[1])}, min_samples_leaf={int(best_params_dtc[2])}"
)
print(f"Best CV error: {best_error_dtc:.4f}")

# Train final DTC model and evaluate on test set
dtc_final = DecisionTreeClassifier(
    max_depth=int(best_params_dtc[0]),
    min_samples_split=int(best_params_dtc[1]),
    min_samples_leaf=int(best_params_dtc[2]),
    random_state=42
)
dtc_final.fit(X_train, y_train)
y_pred_dtc = dtc_final.predict(X_test)
test_acc_dtc = accuracy_score(y_test, y_pred_dtc)
print(f"Test accuracy for DTC: {test_acc_dtc:.4f}\n")

# Optimize ETC using BOA
print("Optimizing ETC with BOA...")
best_params_etc, best_error_etc = boa_optimize(
    objective_etc, lb_etc, ub_etc, N=20, T=50
)
print(
    f"Best ETC params: n_estimators={int(best_params_etc[0])}, max_depth={int(best_params_etc[1])}, min_samples_split={int(best_params_etc[2])}, min_samples_leaf={int(best_params_etc[3])}"
)
print(f"Best CV error: {best_error_etc:.4f}")

# Train final ETC model and evaluate on test set
etc_final = ExtraTreesClassifier(
    n_estimators=int(best_params_etc[0]),
    max_depth=int(best_params_etc[1]),
    min_samples_split=int(best_params_etc[2]),
    min_samples_leaf=int(best_params_etc[3]),
    random_state=42
)
etc_final.fit(X_train, y_train)
y_pred_etc = etc_final.predict(X_test)
test_acc_etc = accuracy_score(y_test, y_pred_etc)
print(f"Test accuracy for ETC: {test_acc_etc:.4f}\n")

# Optimize XGBC using BOA
print("Optimizing XGBC with BOA...")
best_params_xgbc, best_error_xgbc = boa_optimize(
    objective_xgbc, lb_xgbc, ub_xgbc, N=20, T=50
)
print(
    f"Best XGBC params: max_depth={int(best_params_xgbc[0])}, learning_rate={best_params_xgbc[1]:.4f}, n_estimators={int(best_params_xgbc[2])}, subsample={best_params_xgbc[3]:.4f}"
)
print(f"Best CV error: {best_error_xgbc:.4f}")

# Train final XGBC model and evaluate on test set
xgbc_final = XGBClassifier(
    max_depth=int(best_params_xgbc[0]),
    learning_rate=best_params_xgbc[1],
    n_estimators=int(best_params_xgbc[2]),
    subsample=best_params_xgbc[3],
    random_state=42
)
xgbc_final.fit(X_train, y_train)
y_pred_xgbc = xgbc_final.predict(X_test)
test_acc_xgbc = accuracy_score(y_test, y_pred_xgbc)
print(f"Test accuracy for XGBC: {test_acc_xgbc:.4f}")