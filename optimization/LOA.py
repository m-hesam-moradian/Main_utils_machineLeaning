# =====================================================
# 0. IMPORTS
# =====================================================
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
SHEET_NAME = "Data_after_KFold_LSSVC"

CONFIG = {
    "optimizer": "LOA",      # Lyrebird Optimization Algorithm
    "population": 25,
    "iterations": 100,
    "cv": 5,
    "random_state": 42,
    "test_size": 0.2
}

# =====================================================
# 2. MODEL DEFINITION (ETC)
# =====================================================
MODEL = {
    "name": "ETC",
    "builder": ExtraTreesRegressor,
    "bounds": {
        "n_estimators": (100, 800, int),
        "max_depth": (5, 50, int),
        "min_samples_split": (2, 20, int),
        "min_samples_leaf": (1, 10, int),
        "max_features": (0.3, 1.0, float)
    }
}

# =====================================================
# 3. DATA LOADING & PREPROCESSING
# =====================================================
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"]
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# 4. PARAMETER DECODER
# =====================================================
def decode_params(vec, bounds, cast):
    params = {}
    for i, key in enumerate(bounds.keys()):
        low, high, dtype = bounds[key]
        val = vec[i]
        if dtype == int:
            params[key] = int(np.round(val))
        else:
            params[key] = float(val)
    return params

# =====================================================
# 5. OBJECTIVE FUNCTION (MSE)
# =====================================================
def make_objective(model_builder, bounds, cast):
    def objective(vec):
        params = decode_params(vec, bounds, cast)
        model = model_builder(
            **params,
            random_state=CONFIG["random_state"],
            n_jobs=-1
        )
        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CONFIG["cv"],
            scoring="neg_mean_squared_error",
            n_jobs=-1
        ).mean()
        return -score  # minimize MSE
    return objective

# =====================================================
# 6. LYREBIRD OPTIMIZATION ALGORITHM (LOA)
# =====================================================
def LOA(objective, lb, ub, N, T, cast):
    """
    Lyrebird Optimization Algorithm (LOA)
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
        alpha = 1 - (t / T)

        for i in range(N):
            r = np.random.rand()

            if r < 0.4:
                # Imitation (exploitation)
                pop[i] = pop[i] + alpha * np.random.rand(D) * (best_pos - pop[i])

            elif r < 0.7:
                # Exploration
                rand = lb + np.random.rand(D) * (ub - lb)
                pop[i] = pop[i] + np.random.rand() * (rand - pop[i])

            else:
                # Local fine tuning
                pop[i] = best_pos + np.random.normal(0, 0.1, D)

            pop[i] = np.clip(pop[i], lb, ub)
            f = objective(pop[i])

            if f < fit[i]:
                fit[i] = f
                if f < best_fit:
                    best_fit = f
                    best_pos = pop[i].copy()

        best_decoded = decode_params(best_pos, MODEL["bounds"], cast)
        log.append(
            [t + 1] +
            [best_decoded[k] for k in MODEL["bounds"].keys()] +
            [best_fit]
        )

        print(f"Iter {t+1:03d} | Best MSE: {best_fit:.6f}")

    runtime = time.time() - start
    return best_decoded, best_fit, runtime, log

# =====================================================
# 7. OPTIMIZATION EXECUTION
# =====================================================
lb = np.array([v[0] for v in MODEL["bounds"].values()])
ub = np.array([v[1] for v in MODEL["bounds"].values()])
cast = [v[2] for v in MODEL["bounds"].values()]

objective = make_objective(MODEL["builder"], MODEL["bounds"], cast)

best_params, best_mse, runtime, history = LOA(
    objective,
    lb,
    ub,
    N=CONFIG["population"],
    T=CONFIG["iterations"],
    cast=cast
)

print("\nBest Parameters:")
print(best_params)
print("Best CV MSE:", best_mse)
print("Optimization Time (s):", runtime)

# =====================================================
# 8. FINAL MODEL TRAINING & TESTING
# =====================================================
final_model = ExtraTreesRegressor(
    **best_params,
    random_state=CONFIG["random_state"],
    n_jobs=-1
)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\nTest Set Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))
