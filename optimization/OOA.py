# =====================================================
# 0. IMPORTS
# =====================================================
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from catboost import CatBoostRegressor

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
SHEET_NAME = "Data_after_KFold_LSSVC"

CONFIG = {
    "optimizer": "OOA",      # Orangutan Optimization Algorithm
    "population": 25,
    "iterations": 100,
    "cv": 5,
    "random_state": 42,
    "test_size": 0.2
}

# =====================================================
# 2. MODEL DEFINITION (CatBoost)
# =====================================================
MODEL = {
    "name": "CatBoost",
    "builder": CatBoostRegressor,
    "bounds": {
        "iterations": (200, 1500, int),
        "depth": (4, 12, int),
        "learning_rate": (0.01, 0.3, float),
        "l2_leaf_reg": (1, 10, float),
        "bagging_temperature": (0, 5, float)
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
def decode_params(vec, bounds):
    params = {}
    for i, key in enumerate(bounds.keys()):
        low, high, dtype = bounds[key]
        if dtype == int:
            params[key] = int(np.round(vec[i]))
        else:
            params[key] = float(vec[i])
    return params

# =====================================================
# 5. OBJECTIVE FUNCTION (MSE)
# =====================================================
def make_objective(model_builder, bounds):
    def objective(vec):
        params = decode_params(vec, bounds)

        model = model_builder(
            **params,
            random_seed=CONFIG["random_state"],
            verbose=0
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
# 6. ORANGUTAN OPTIMIZATION ALGORITHM (OOA)
# =====================================================
def OOA(objective, lb, ub, N, T):
    """
    Orangutan Optimization Algorithm (OOA)
    """
    start = time.time()
    D = len(lb)

    pop = lb + np.random.rand(N, D) * (ub - lb)
    fitness = np.array([objective(pop[i]) for i in range(N)])

    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    log = []

    for t in range(T):

        for i in range(N):

            r1 = np.random.rand()
            r2 = np.random.rand()

            if r1 < 0.5:
                # 🌴 Swinging movement (exploration)
                new_pos = pop[i] + r2 * (np.random.rand(D) * (ub - lb) - pop[i])
            else:
                # 🐵 Social climbing (exploitation toward best)
                new_pos = pop[i] + r2 * (best_pos - pop[i])

            new_pos = np.clip(new_pos, lb, ub)
            new_fit = objective(new_pos)

            if new_fit < fitness[i]:
                pop[i] = new_pos
                fitness[i] = new_fit

                if new_fit < best_fit:
                    best_fit = new_fit
                    best_pos = new_pos.copy()

        best_decoded = decode_params(best_pos, MODEL["bounds"])
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

objective = make_objective(MODEL["builder"], MODEL["bounds"])

best_params, best_mse, runtime, history = OOA(
    objective,
    lb,
    ub,
    N=CONFIG["population"],
    T=CONFIG["iterations"]
)

print("\nBest Parameters:")
print(best_params)
print("Best CV MSE:", best_mse)
print("Optimization Time (s):", runtime)

# =====================================================
# 8. FINAL MODEL TRAINING & TESTING
# =====================================================
final_model = CatBoostRegressor(
    **best_params,
    random_seed=CONFIG["random_state"],
    verbose=0
)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\nTest Set Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))
