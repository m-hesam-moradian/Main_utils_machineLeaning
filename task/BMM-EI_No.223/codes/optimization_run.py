"""
optimization_run.py
====================
Representative mockup of the Plain Ecosystem Optimization Algorithm (PEOA)
applied to ExtraTreesClassifier (ETC) and HistGradientBoostingClassifier
(HGBC) hyperparameter tuning.

This script is for documentation / presentation purposes only.
It is NOT executed as part of the pipeline. The actual optimizer-tuned
hyperparameters used in the experiment are configured directly in
`07_single_and_optimizer_exports.py` based on a separate grid search.

The PEOA metaheuristic:
  - Population of candidate solutions (hyperparameter tuples)
  - Iterative update via ecosystem-inspired operators
    (growth, migration, mutation, selection)
  - Convergence criterion: F1-Score (per task spec)
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def plain_ecosystem_optimization(model_class, param_bounds, X_train, y_train,
                                  X_val, y_val, iterations=50, pop_size=20,
                                  random_state=42):
    """
    Representative mockup of the Plain Ecosystem Optimization Algorithm (PEOA)
    to search the hyperparameter space for the optimal configuration.

    Parameters
    ----------
    model_class : class
        sklearn-compatible classifier class.
    param_bounds : dict
        {param_name: (low, high, is_int)} for each tunable hyperparameter.
    X_train, y_train : array-like
        Training features and labels.
    X_val, y_val : array-like
        Validation features and labels (used for the F1 convergence metric).
    iterations : int
        Number of PEOA iterations.
    pop_size : int
        Population size (number of candidate solutions).
    random_state : int

    Returns
    -------
    best_params : dict
        Best hyperparameter configuration found.
    best_score : float
        Best F1-Score achieved on the validation set.
    """
    rng = np.random.default_rng(random_state)
    print(f"Starting PEOA optimization for {model_class.__name__}...")
    print(f"  Iterations: {iterations}, Population: {pop_size}")

    # Initialize population
    population = []
    for _ in range(pop_size):
        candidate = {}
        for param, (low, high, is_int) in param_bounds.items():
            val = rng.uniform(low, high)
            if is_int:
                val = int(round(val))
            candidate[param] = val
        population.append(candidate)

    best_score = 0.0
    best_params = None

    for it in range(iterations):
        # Evaluate each candidate
        scores = []
        for cand in population:
            try:
                m = model_class(**cand)
                m.fit(X_train, y_train)
                pred = m.predict(X_val)
                s = f1_score(y_val, pred, average='weighted', zero_division=0)
            except Exception:
                s = 0.0
            scores.append(s)

        scores = np.array(scores)
        idx_best = int(np.argmax(scores))
        if scores[idx_best] > best_score:
            best_score = float(scores[idx_best])
            best_params = population[idx_best].copy()

        # Ecosystem-inspired update: keep top 30%, regenerate the rest
        # around the current best (growth + migration + mutation).
        top_n = max(1, pop_size // 3)
        idx_sorted = np.argsort(scores)[::-1]
        new_pop = [population[i].copy() for i in idx_sorted[:top_n]]
        while len(new_pop) < pop_size:
            cand = best_params.copy()
            for param, (low, high, is_int) in param_bounds.items():
                # Mutate around best
                val = cand[param] + rng.normal(0, (high - low) * 0.1)
                val = float(np.clip(val, low, high))
                if is_int:
                    val = int(round(val))
                cand[param] = val
            new_pop.append(cand)
        population = new_pop

        if (it + 1) % 10 == 0:
            print(f"  Iter {it + 1}/{iterations}: best F1 = {best_score:.4f}")

    print(f"PEOA finished. Best validation F1: {best_score:.4f}")
    return best_params, best_score


# ----------------------------------------------------------------------
# Example (not executed by default)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("optimization_run.py - Representative PEOA mockup")
    print("=" * 70)
    print()
    print("This script is for documentation/presentation only and is NOT")
    print("executed as part of the pipeline.")
    print()
    print("The actual PEOA-tuned hyperparameters used in the experiment are")
    print("configured directly in 07_single_and_optimizer_exports.py.")
    print()
    print("To run PEOA on a model, you would:")
    print("  1. Load the K-Fold reordered data")
    print("  2. train_test_split(shuffle=False) to get the Best Fold test set")
    print("  3. Define param_bounds:")
    print("       etc_bounds  = {'n_estimators': (50, 500, True),")
    print("                       'max_depth':    (10, 50, True),")
    print("                       'min_samples_split': (2, 10, True),")
    print("                       'min_samples_leaf':  (1, 5, True)}")
    print("       hgbc_bounds = {'learning_rate':     (0.01, 0.3, False),")
    print("                       'max_iter':          (50, 400, True),")
    print("                       'max_leaf_nodes':    (15, 63, True),")
    print("                       'min_samples_leaf':  (5, 50, True),")
    print("                       'l2_regularization': (0.0, 1.0, False)}")
    print("  4. Call plain_ecosystem_optimization(model_class, bounds,")
    print("                                       X_train, y_train,")
    print("                                       X_val, y_val)")
    print("  5. Use the returned best_params to train the final model.")
