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


# Haze Optimization Algorithm (HOA) implementation
def hoa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Haze Optimization Algorithm for minimizing an objective function.

    Inspired by atmospheric haze phenomena: dispersion (spreading particles for exploration),
    scattering (light interaction for local adjustments), and condensation (gathering for exploitation).

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
    # Initialize population
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        # Haze density factor (decreases over time, simulating clearing)
        density = 1 - (t / T)  # From 1 (dense haze) to 0 (clear)

        for i in range(N):
            # Phase 1: Dispersion (Exploration - particle spreading in haze)
            if np.random.rand() < density:  # More dispersion in early iterations
                # Random spread in parameter space
                r1 = np.random.rand(D)
                spread_factor = density * np.random.uniform(0.2, 0.8)
                new_pos = population[i] + spread_factor * (2 * r1 - 1) * (ub - lb) / 5

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

            # Phase 2: Scattering (Local adjustments - interaction with nearby particles)
            # Select a random other particle
            j = np.random.randint(0, N)
            while j == i:
                j = np.random.randint(0, N)

            r2 = np.random.rand(D)
            scatter_factor = density * np.random.uniform(0.1, 0.5)
            new_pos = population[i] + scatter_factor * r2 * (
                population[j] - population[i]
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

            # Phase 3: Condensation (Exploitation - gather towards best in clearing haze)
            if np.random.rand() < (
                1 - density
            ):  # More condensation in later iterations
                r3 = np.random.rand(D)
                condense_step = (1 - density) * np.random.uniform(0.3, 1.0)
                new_pos = population[i] + condense_step * r3 * (
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

# Optimize ETC using HOA
print("Optimizing Extra Trees Classifier (ETC) with HOA...")
best_params_etc, best_score_etc = hoa_optimize(
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

# Optimize GBC using HOA
print("Optimizing Gradient Boosting Classifier (GBC) with HOA...")
best_params_gbc, best_score_gbc = hoa_optimize(
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

# Optimize RFC using HOA
print("Optimizing Random Forest Classifier (RFC) with HOA...")
best_params_rfc, best_score_rfc = hoa_optimize(
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
print("HOA OPTIMIZATION SUMMARY")
print("=" * 60)
models_summary = {"ETC": test_acc_etc, "GBC": test_acc_gbc, "RFC": test_acc_rfc}
best_model = max(models_summary, key=models_summary.get)
print(
    f"Best performing model: {best_model} (Test Accuracy: {models_summary[best_model]:.4f})"
)
for model, acc in models_summary.items():
    print(f"{model:4s}: Test Accuracy = {acc:.4f}")
