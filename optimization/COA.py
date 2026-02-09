import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from ngboost import NGBRegressor
from ngboost.distns import Normal

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_LSSVC"
CONFIG = {
    "optimizer": "COA",  # Cyclone Optimization Algorithm
    "population": 25,
    "iterations": 100,   # NGBoost is slower than SVC, adjusted iters
    "cv": 5,
    "random_state": 42
}

# =====================================================
# 2. MODEL DEFINITION (NGBR)
# =====================================================
MODEL = {
    "name": "NGBR",
    "builder": NGBRegressor,
    "bounds": {
        "n_estimators": (50, 500, int),
        "learning_rate": (0.01, 0.2, float),
        "minibatch_frac": (0.1, 1.0, float)
    }
}

# [Data Loading and Helper Functions remain similar, but target Regression metrics]
# Load data code goes here (StandardScaler etc.)
# ... (use the code from your snippet for X, y and Scaling) ...

def make_objective(model_builder, bounds, cast):
    def objective(vec):
        params = decode_params(vec, bounds, cast)
        # NGBoost uses 'Dist' for probability distribution
        model = model_builder(
            **params,
            Dist=Normal,
            random_state=CONFIG["random_state"],
            verbose=False
        )
        # Using Negative MSE because we minimize the objective
        score = cross_val_score(
            model, X_train, y_train,
            cv=CONFIG["cv"],
            scoring="neg_mean_squared_error",
            n_jobs=-1
        ).mean()
        return -score # Return positive MSE to minimize
    return objective

# =====================================================
# 4. CYCLONE OPTIMIZATION ALGORITHM (COA)
# =====================================================
def COA(objective, lb, ub, N, T, cast):
    """
    Cyclone Optimization Algorithm (COA)
    Simulates the spiraling motion of a cyclone towards the eye (best solution).
    """
    start = time.time()
    D = len(lb)
    pop = lb + np.random.rand(N, D) * (ub - lb)
    fit = np.array([objective(pop[i]) for i in range(N)])

    best_idx = np.argmin(fit)
    best_pos = pop[best_idx].copy()
    best_fit = fit[best_idx]

    log = []
    for t in range(T):
        # Cyclone Intensity Factor (reduces over time)
        a = 2 - t * (2 / T) 
        
        for i in range(N):
            r = np.random.rand()
            # Spiraling behavior (The 'Cyclone' motion)
            if r < 0.5:
                # Search towards the eye (Exploitation)
                dist_to_eye = np.abs(best_pos - pop[i])
                angle = 2 * np.pi * np.random.uniform(-1, 1)
                spiral = dist_to_eye * np.exp(angle) * np.cos(angle)
                pop[i] = best_pos + spiral
            else:
                # Random drift in the cyclone (Exploration)
                rand_idx = np.random.randint(0, N)
                pop[i] = pop[rand_idx] + a * np.random.uniform(-1, 1, D)

            pop[i] = np.clip(pop[i], lb, ub)
            f = objective(pop[i])

            if f < fit[i]:
                fit[i] = f
                if f < best_fit:
                    best_pos, best_fit = pop[i].copy(), f

        best_decoded = decode_params(best_pos, MODEL["bounds"], cast)
        log.append([t + 1] + [best_decoded[k] for k in MODEL["bounds"].keys()] + [best_fit])
        print(f"Iter {t+1:03d} | Best MSE: {best_fit:.4f}")

    return best_decoded, best_fit, time.time() - start, log

# =====================================================
# 5. EXECUTION
# =====================================================
# ... (Initialize lb, ub, objective as in your original script) ...
# results = COA(...)