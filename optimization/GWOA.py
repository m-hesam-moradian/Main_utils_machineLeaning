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
    "optimizer": "GWO",      # Grey Wolf Optimization
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
        _, _, dtype = bounds[key]
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

        return -score
    return objective

# =====================================================
# 6. GREY WOLF OPTIMIZATION (GWO)
# =====================================================
def GWO(objective, lb, ub, N, T):

    start = time.time()
    D = len(lb)

    # Initialize wolves
    pop = lb + np.random.rand(N, D) * (ub - lb)
    fitness = np.array([objective(pop[i]) for i in range(N)])

    # Identify alpha, beta, delta
    sorted_idx = np.argsort(fitness)
    alpha = pop[sorted_idx[0]].copy()
    beta = pop[sorted_idx[1]].copy()
    delta = pop[sorted_idx[2]].copy()

    alpha_score = fitness[sorted_idx[0]]

    log = []

    for t in range(T):

        a = 2 - t * (2 / T)  # linearly decreases from 2 to 0

        for i in range(N):

            for j in range(D):

                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - pop[i, j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - pop[i, j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - pop[i, j])
                X3 = delta[j] - A3 * D_delta

                pop[i, j] = (X1 + X2 + X3) / 3

            pop[i] = np.clip(pop[i], lb, ub)

        # Evaluate new population
        fitness = np.array([objective(pop[i]) for i in range(N)])
        sorted_idx = np.argsort(fitness)

        alpha = pop[sorted_idx[0]].copy()
        beta = pop[sorted_idx[1]].copy()
        delta = pop[sorted_idx[2]].copy()

        alpha_score = fitness[sorted_idx[0]]

        best_decoded = decode_params(alpha, MODEL["bounds"])
        log.append(
            [t + 1] +
            [best_decoded[k] for k in MODEL["bounds"].keys()] +
            [alpha_score]
        )

        print(f"Iter {t+1:03d} | Best MSE: {alpha_score:.6f}")

    runtime = time.time() - start
    return best_decoded, alpha_score, runtime, log

# =====================================================
# 7. OPTIMIZATION EXECUTION
# =====================================================
lb = np.array([v[0] for v in MODEL["bounds"].values()])
ub = np.array([v[1] for v in MODEL["bounds"].values()])

objective = make_objective(MODEL["builder"], MODEL["bounds"])

best_params, best_mse, runtime, history = GWO(
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
