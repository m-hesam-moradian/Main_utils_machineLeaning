import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from hpelm import ELM
from catboost import CatBoostClassifier
from scipy.special import gamma

# Load sample dataset for demonstration (treat as binary classification)
data = load_diabetes()
X, y = data.data, data.target
y_bin = (y > np.median(y)).astype(int)  # Binarize for classification
X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.2, random_state=42
)


# Levy flight function
def levy_flight(D, beta=1.5):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(D) * sigma
    v = np.random.randn(D)
    step = u / np.abs(v) ** (1 / beta)
    return 0.01 * step


# Arctic Puffin Optimization (APO) implementation
def apo_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Arctic Puffin Optimization Algorithm for minimizing an objective function.
    Inspired by aerial flight (exploration) and underwater foraging (exploitation) of Arctic puffins.

    Parameters:
    - objective_func: function that takes a 1D array of parameters and returns a scalar to minimize (negative accuracy).
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
        # Behavioral conversion factor B (for phase transition)
        B = 2 * np.log(1 / np.random.rand()) * (1 - t / T)

        for i in range(N):
            r = np.random.randint(0, N)  # Random other puffin
            if np.random.rand() < B:  # Exploration: Aerial flight phase
                # Aerial search strategy
                L = levy_flight(D)
                alpha = 2 * (1 - t / T)  # Adaptive velocity factor
                R = np.round(0.5 * (0.05 + np.random.rand())) * alpha
                new_pos = population[i] + (population[i] - population[r]) * L + R

                # Swoop predation (plunge dive toward best, simplified)
                if np.random.rand() < 0.5:
                    synergy = np.random.rand(D)  # Synergy factor
                    new_pos = new_pos + synergy * (best_params - new_pos)

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
                        print(f"Iteration {t}: Best params = {best_params}, Best score = {best_score:.4f}")
            else:  # Exploitation: Underwater foraging phase
                # Gathering foraging (converge to best)
                alpha_change = np.random.uniform(0.5, 1.5, D)  # Adaptive change factors
                new_pos = population[i] + alpha_change * (best_params - population[i])

                # Intensified search (local perturbation)
                beta = 2 * np.random.rand(D)
                rand_pos = population[r]
                new_pos += beta * np.random.rand() * (population[i] - rand_pos)

                # Predator evasion (small random escape if needed)
                if np.random.rand() < 0.1:
                    evasion = np.random.uniform(-0.1, 0.1, D) * (ub - lb)
                    new_pos += evasion

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
                        print(f"Iteration {t}: Best params = {best_params}, Best score = {best_score:.4f}")

    return best_params, best_score


# Objective function for Extreme Learning Machine (ELM)
def objective_elm(params):
    """
    Objective for ELM classifier.
    Params: [n_hidden, C]
    """
    n_hidden = int(params[0])
    C = params[1]

    model = ELM(n_hidden=n_hidden, C=C)

    # Use cross-validation score (accuracy, but return negative for minimization)
    score = cross_val_score(
        model, X_train, y_train_bin, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score  # Minimize error (1 - accuracy)


# Objective function for CatBoost Classifier
def objective_catboost(params):
    """
    Objective for CatBoostClassifier.
    Params: [depth, learning_rate, iterations]
    """
    depth = int(params[0])
    lr = params[1]
    iterations = int(params[2])

    model = CatBoostClassifier(
        depth=depth,
        learning_rate=lr,
        iterations=iterations,
        random_state=42,
        verbose=0,  # Suppress output
    )

    # Use cross-validation score (accuracy, negative for minimization)
    score = cross_val_score(
        model, X_train, y_train_bin, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score  # Minimize error


# Hyperparameter bounds for ELM
lb_elm = np.array([10, 0.1])
ub_elm = np.array([100, 10.0])

# Hyperparameter bounds for CatBoost
lb_cat = np.array([3, 0.01, 50])
ub_cat = np.array([10, 0.3, 200])


# Optimize ELM using APO
print("Optimizing ELM with APO...")
best_params_elm, best_score_elm = apo_optimize(
    objective_elm, lb_elm, ub_elm, N=20, T=50
)
print(f"Best ELM params: n_hidden={int(best_params_elm[0])}, C={best_params_elm[1]:.4f}")
print(f"Best CV Error: {best_score_elm:.4f}")

# Train final ELM model and evaluate on test set
elm_final = ELM(n_hidden=int(best_params_elm[0]), C=best_params_elm[1])
elm_final.fit(X_train, y_train_bin)
y_pred_elm = elm_final.predict(X_test)
test_acc_elm = accuracy_score(y_test_bin, y_pred_elm)
print(f"Test Accuracy for ELM: {test_acc_elm:.4f}\n")


# Optimize CatBoost using APO
print("Optimizing CatBoost with APO...")
best_params_cat, best_score_cat = apo_optimize(
    objective_cat, lb_cat, ub_cat, N=20, T=50
)
print(f"Best CatBoost params: depth={int(best_params_cat[0])}, learning_rate={best_params_cat[1]:.4f}, iterations={int(best_params_cat[2])}")
print(f"Best CV Error: {best_score_cat:.4f}")

# Train final CatBoost model and evaluate on test set
cat_final = CatBoostClassifier(
    depth=int(best_params_cat[0]),
    learning_rate=best_params_cat[1],
    iterations=int(best_params_cat[2]),
    random_state=42,
    verbose=0,
)
cat_final.fit(X_train, y_train_bin)
y_pred_cat = cat_final.predict(X_test)
test_acc_cat = accuracy_score(y_test_bin, y_pred_cat)
print(f"Test Accuracy for CatBoost: {test_acc_cat:.4f}")