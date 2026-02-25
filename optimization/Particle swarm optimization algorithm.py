import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# =====================================================
# 1. CONFIGURATION
# =====================================================
CONFIG = {
    "optimizer": "PSO",
    "population": 25,
    "iterations": 50,
    "cv": 5,
    "random_state": 42,

    # PSO parameters
    "w": 0.7,      # inertia weight
    "c1": 1.5,     # cognitive coefficient
    "c2": 1.5      # social coefficient
}

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
# 4. PARTICLE SWARM OPTIMIZATION (PSO)
# =====================================================
def PSO(objective, lb, ub, N, T):

    start = time.time()
    D = len(lb)

    # Initialize particles
    pos = np.random.uniform(0, 1, (N, D))
    vel = np.zeros((N, D))

    # Personal best
    pbest_pos = pos.copy()
    pbest_fit = np.array([objective(p) for p in pos])

    # Global best
    gbest_index = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_index].copy()
    gbest_fit = pbest_fit[gbest_index]

    log = []

    w = CONFIG["w"]
    c1 = CONFIG["c1"]
    c2 = CONFIG["c2"]

    for t in range(T):

        for i in range(N):

            r1 = np.random.rand(D)
            r2 = np.random.rand(D)

            # Velocity update
            vel[i] = (
                w * vel[i]
                + c1 * r1 * (pbest_pos[i] - pos[i])
                + c2 * r2 * (gbest_pos - pos[i])
            )

            # Position update
            pos[i] = pos[i] + vel[i]
            pos[i] = np.clip(pos[i], 0, 1)

            fitness = objective(pos[i])

            # Update personal best
            if fitness < pbest_fit[i]:
                pbest_fit[i] = fitness
                pbest_pos[i] = pos[i].copy()

        # Update global best
        best_idx = np.argmin(pbest_fit)
        if pbest_fit[best_idx] < gbest_fit:
            gbest_fit = pbest_fit[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()

        best_decoded = decode_params(gbest_pos, MODEL["bounds"])
        log.append([t + 1, -gbest_fit])

        print(f"Iter {t+1:03d} | Best Accuracy: {-gbest_fit:.4f}")

    return best_decoded, -gbest_fit, time.time() - start, log


# =====================================================
# RUN OPTIMIZATION
# =====================================================
objective = make_objective(MODEL["builder"], MODEL["bounds"])

lb = np.zeros(len(MODEL["bounds"]))
ub = np.ones(len(MODEL["bounds"]))

best_params, best_score, runtime, history = PSO(
    objective,
    lb,
    ub,
    CONFIG["population"],
    CONFIG["iterations"]
)

print("\nBest Parameters:", best_params)
print("Best Accuracy:", best_score)
print("Runtime:", runtime)