import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# =====================================================
# 1. CONFIGURATION
# =====================================================
CONFIG = {
    "optimizer": "ACO",
    "population": 25,     # number of ants
    "iterations": 50,
    "cv": 5,
    "random_state": 42,

    # ACO parameters
    "evaporation": 0.3,   # pheromone evaporation rate
    "alpha": 1.0,         # pheromone importance
    "sigma": 0.2          # sampling spread
}

np.random.seed(CONFIG["random_state"])

# =====================================================
# 2. MODEL DEFINITION (XGBoost Classifier)
# =====================================================
MODEL = {
    "name": "XGBC",
    "builder": XGBClassifier,
    "bounds": {
        "n_estimators": (10, 300, int),
        "learning_rate": (0.001, 0.3, float),
        "max_depth": (2, 12, int),
    }
}

# =====================================================
# PARAMETER DECODER
# =====================================================
def decode_params(vec, bounds):
    params = {}
    for i, (key, (low, high, cast)) in enumerate(bounds.items()):
        val = low + vec[i] * (high - low)
        params[key] = cast(val)
    return params


# =====================================================
# OBJECTIVE FUNCTION
# =====================================================
def make_objective(model_builder, bounds):

    def objective(vec):

        params = decode_params(vec, bounds)

        model = model_builder(
            **params,
            random_state=CONFIG["random_state"],
            eval_metric="logloss",
            use_label_encoder=False
        )

        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CONFIG["cv"],
            scoring="accuracy",
            n_jobs=-1
        ).mean()

        return -score  # minimize

    return objective


# =====================================================
# 3. ANT COLONY OPTIMIZATION (Continuous ACO)
# =====================================================
def ACO(objective, lb, ub, N, T):

    start = time.time()
    D = len(lb)

    # Initialize pheromone center (mean solution)
    pheromone_mean = np.random.uniform(0, 1, D)
    pheromone_std = np.ones(D) * CONFIG["sigma"]

    best_pos = None
    best_fit = np.inf

    log = []

    evaporation = CONFIG["evaporation"]
    alpha = CONFIG["alpha"]

    for t in range(T):

        solutions = []
        fitnesses = []

        # ============================
        # Ants construct solutions
        # ============================
        for i in range(N):

            # Sample around pheromone distribution
            candidate = np.random.normal(
                pheromone_mean,
                pheromone_std
            )

            candidate = np.clip(candidate, 0, 1)

            fit = objective(candidate)

            solutions.append(candidate)
            fitnesses.append(fit)

            if fit < best_fit:
                best_fit = fit
                best_pos = candidate.copy()

        solutions = np.array(solutions)
        fitnesses = np.array(fitnesses)

        # ============================
        # Pheromone Update
        # ============================

        # Convert fitness → quality (higher is better)
        quality = 1 / (fitnesses - fitnesses.min() + 1e-10)

        weights = quality ** alpha
        weights /= weights.sum()

        # Update pheromone mean (weighted solutions)
        new_mean = np.sum(solutions * weights[:, None], axis=0)

        # Evaporation + reinforcement
        pheromone_mean = (
            (1 - evaporation) * pheromone_mean
            + evaporation * new_mean
        )

        # Reduce exploration over time
        pheromone_std *= 0.97

        log.append([t + 1, -best_fit])

        print(f"Iter {t+1:03d} | Best Accuracy: {-best_fit:.4f}")

    best_decoded = decode_params(best_pos, MODEL["bounds"])

    return best_decoded, -best_fit, time.time() - start, log


# =====================================================
# RUN OPTIMIZATION
# =====================================================
objective = make_objective(MODEL["builder"], MODEL["bounds"])

lb = np.zeros(len(MODEL["bounds"]))
ub = np.ones(len(MODEL["bounds"]))

best_params, best_score, runtime, history = ACO(
    objective,
    lb,
    ub,
    CONFIG["population"],
    CONFIG["iterations"]
)

print("\nBest Parameters:", best_params)
print("Best Accuracy:", best_score)
print("Runtime:", runtime)