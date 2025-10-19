import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score

# Load sample classification dataset for demonstration
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Echidna Optimization Algorithm (EOA) implementation
def eoa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Echidna Optimization Algorithm for minimizing an objective function.

    Inspired by echidna behavior: spiny defense (protecting good solutions by avoiding poor areas),
    tongue foraging (extending tongue for exploration to catch prey), and burrowing (digging deep for exploitation and hibernation-like convergence).

    Parameters:
    - objective_func: function that takes a 1D array of parameters and returns a scalar value to minimize.
    - lb: lower bounds (array of length D)
    - ub: upper bounds (array of length D)
    - N: population size
    - T: number of iterations

    Returns:
    - best_params: optimized parameters
    - best_score: best objective value
    """
    D = len(lb)
    # Initialize population (echidnas in the landscape)
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        # Adaptation factor (decreases over time, like seasonal changes affecting behavior)
        adapt = 2 * (1 - t / T)  # From 2 to 0

        for i in range(N):
            # Phase 1: Spiny Defense (Avoid poor areas - repel from worst solutions)
            if np.random.rand() < 0.4:  # Probabilistic defense
                # Find the worst individual
                worst_idx = np.argmax(fitness)
                r1 = np.random.rand(D)
                repel_factor = adapt * np.random.uniform(0.1, 0.5)
                new_pos = population[i] + repel_factor * r1 * (
                    population[i] - population[worst_idx]
                )

                # Clip to bounds
                new_pos = np.clip(new_pos, lb, ub)

                # Evaluate
                new_fit = objective_func(new_pos)
                if new_fit < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fit
                    if new_fit < best_score:
                        best_params = new_pos.copy()
                        best_score = new_fit

            # Phase 2: Tongue Foraging (Exploration - extend tongue to probe nearby areas)
            r2 = np.random.rand(D)
            probe_step = adapt * (2 * r2 - 1) * (ub - lb) / 10  # Sticky tongue probe
            new_pos = population[i] + probe_step

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit

            # Phase 3: Burrowing (Exploitation - dig towards best position for safety/prey)
            if np.random.rand() < 0.6:  # Higher chance for exploitation
                r3 = np.random.rand(D)
                dig_factor = (1 - adapt / 2) * np.random.uniform(
                    0.2, 0.8
                )  # Increases over time
                new_pos = population[i] + dig_factor * r3 * (
                    best_params - population[i]
                )

                # Clip to bounds
                new_pos = np.clip(new_pos, lb, ub)

                # Evaluate
                new_fit = objective_func(new_pos)
                if new_fit < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fit
                    if new_fit < best_score:
                        best_params = new_pos.copy()
                        best_score = new_fit

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_score:
            best_params = population[best_idx].copy()
            best_score = fitness[best_idx]

    return best_params, best_score


# Objective function for Extra Trees Classifier (ETC)
def objective_etc(params):
    """Objective for ExtraTreesClassifier.
    Params: [n_estimators, max_depth, min_samples_split]
    """
    n_est = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])

    model = ExtraTreesClassifier(
        n_estimators=n_est,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return -score  # Minimize negative accuracy


# Objective function for Gradient Boosting Classifier (GBC)
def objective_gbc(params):
    """Objective for GradientBoostingClassifier.
    Params: [n_estimators, learning_rate, max_depth]
    """
    n_est = int(params[0])
    lr = params[1]
    max_depth = int(params[2])

    model = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=lr, max_depth=max_depth, random_state=42
    )

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return -score


# Objective function for Random Forest Classifier (RFC)
def objective_rfc(params):
    """Objective for RandomForestClassifier.
    Params: [n_estimators, max_depth, min_samples_split]
    """
    n_est = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])

    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return -score


# Hyperparameter bounds
# ETC bounds
lb_etc = np.array([50, 3, 2])
ub_etc = np.array([200, 20, 10])

# GBC bounds
lb_gbc = np.array([50, 0.01, 3])
ub_gbc = np.array([200, 0.3, 10])

# RFC bounds
lb_rfc = np.array([50, 3, 2])
ub_rfc = np.array([200, 20, 10])

# Optimize ETC using EOA
print("Optimizing Extra Trees Classifier (ETC) with EOA...")
best_params_etc, best_score_etc = eoa_optimize(
    objective_etc, lb_etc, ub_etc, N=20, T=50
)
print(
    f"Best ETC params: n_estimators={int(best_params_etc[0])}, "
    f"max_depth={int(best_params_etc[1]) if best_params_etc[1] > 0 else 'None'}, min_samples_split={int(best_params_etc[2])}"
)
print(
    f"Best CV Negative Accuracy: {best_score_etc:.4f} (Accuracy: {-best_score_etc:.4f})"
)

# Train final ETC model
etc_final = ExtraTreesClassifier(
    n_estimators=int(best_params_etc[0]),
    max_depth=int(best_params_etc[1]) if best_params_etc[1] > 0 else None,
    min_samples_split=int(best_params_etc[2]),
    random_state=42,
)
etc_final.fit(X_train, y_train)
y_pred_etc = etc_final.predict(X_test)
test_acc_etc = accuracy_score(y_test, y_pred_etc)
print(f"Test Accuracy for ETC: {test_acc_etc:.4f}\n")

# Optimize GBC using EOA
print("Optimizing Gradient Boosting Classifier (GBC) with EOA...")
best_params_gbc, best_score_gbc = eoa_optimize(
    objective_gbc, lb_gbc, ub_gbc, N=20, T=50
)
print(
    f"Best GBC params: n_estimators={int(best_params_gbc[0])}, "
    f"learning_rate={best_params_gbc[1]:.4f}, max_depth={int(best_params_gbc[2])}"
)
print(
    f"Best CV Negative Accuracy: {best_score_gbc:.4f} (Accuracy: {-best_score_gbc:.4f})"
)

# Train final GBC model
gbc_final = GradientBoostingClassifier(
    n_estimators=int(best_params_gbc[0]),
    learning_rate=best_params_gbc[1],
    max_depth=int(best_params_gbc[2]),
    random_state=42,
)
gbc_final.fit(X_train, y_train)
y_pred_gbc = gbc_final.predict(X_test)
test_acc_gbc = accuracy_score(y_test, y_pred_gbc)
print(f"Test Accuracy for GBC: {test_acc_gbc:.4f}\n")

# Optimize RFC using EOA
print("Optimizing Random Forest Classifier (RFC) with EOA...")
best_params_rfc, best_score_rfc = eoa_optimize(
    objective_rfc, lb_rfc, ub_rfc, N=20, T=50
)
print(
    f"Best RFC params: n_estimators={int(best_params_rfc[0])}, "
    f"max_depth={int(best_params_rfc[1]) if best_params_rfc[1] > 0 else 'None'}, min_samples_split={int(best_params_rfc[2])}"
)
print(
    f"Best CV Negative Accuracy: {best_score_rfc:.4f} (Accuracy: {-best_score_rfc:.4f})"
)

# Train final RFC model
rfc_final = RandomForestClassifier(
    n_estimators=int(best_params_rfc[0]),
    max_depth=int(best_params_rfc[1]) if best_params_rfc[1] > 0 else None,
    min_samples_split=int(best_params_rfc[2]),
    random_state=42,
)
rfc_final.fit(X_train, y_train)
y_pred_rfc = rfc_final.predict(X_test)
test_acc_rfc = accuracy_score(y_test, y_pred_rfc)
print(f"Test Accuracy for RFC: {test_acc_rfc:.4f}")

# Summary comparison
print("\n" + "=" * 60)
print("EOA OPTIMIZATION SUMMARY")
print("=" * 60)
models_summary = {"ETC": test_acc_etc, "GBC": test_acc_gbc, "RFC": test_acc_rfc}
best_model = max(models_summary, key=models_summary.get)
print(
    f"Best performing model: {best_model} (Test Accuracy: {models_summary[best_model]:.4f})"
)
for model, acc in models_summary.items():
    print(f"{model:4s}: Test Accuracy = {acc:.4f}")
