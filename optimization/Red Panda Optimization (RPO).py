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
    "optimizer": "RPO",      # RedPanda Optimization
    "population": 25,
    "iterations": 100,
    "cv": 5,
    "random_state": 42,
    "test_size": 0.2
}

# =====================================================
# 2. MODEL DEFINITION (Extra Trees Regressor)
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
def decode_params(vec, bounds):
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
def make_objective(model_builder, bounds):
    def objective(vec):
        params = decode_params(vec, bounds)
        model = model_builder(
            **params,
            random_state=CONFIG["random_state"],
            n_jobs=-1
        )
        # We use cross-validation to find the best generalizable params
        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CONFIG["cv"],
            scoring="neg_mean_squared_error",
            n_jobs=-1
        ).mean()
        return -score  # Minimize MSE
    return objective

# =====================================================
# 6. REDPANDA OPTIMIZATION ALGORITHM (RPO)
# =====================================================
def RPO(objective, lb, ub, N, T):
    """
    RedPanda Optimization (RPO)
    Phases: Climbing (Exploration), Scent Marking (Guidance), Resting (Exploitation)
    """
    start = time.time()
    D = len(lb)

    # Initialize Population
    pop = lb + np.random.rand(N, D) * (ub - lb)
    fit = np.array([objective(pop[i]) for i in range(N)])

    best_idx = np.argmin(fit)
    best_pos = pop[best_idx].copy()
    best_fit = fit[best_idx]

    log = []

    for t in range(T):
        # Energy factor decreases to transition from exploration to exploitation
        E = 2 * (1 - (t / T)) 
        
        for i in range(N):
            r = np.random.rand()

            if r < 0.45:
                # PHASE 1: CLIMBING (Exploration)
                climb_factor = np.random.uniform(-1, 1, D)
                rand_idx = np.random.randint(0, N)
                pop[i] = pop[i] + climb_factor * E * (pop[rand_idx] - pop[i])

            elif r < 0.85:
                # PHASE 2: SCENT MARKING (Social learning from best panda)
                scent_guidance = np.random.rand(D) * (best_pos - pop[i])
                pop[i] = pop[i] + scent_guidance

            else:
                # PHASE 3: RESTING (Local search around the current best)
                pop[i] = best_pos + (np.random.normal(0, 1, D) * (ub - lb) * 0.005)

            # Keep pandas within the forest boundaries
            pop[i] = np.clip(pop[i], lb, ub)
            
            # Evaluate new position
            f = objective(pop[i])

            # Greedy Update
            if f < fit[i]:
                fit[i] = f
                if f < best_fit:
                    best_fit = f
                    best_pos = pop[i].copy()

        # Logging iteration results
        current_best_params = decode_params(best_pos, MODEL["bounds"])
        log.append([t + 1] + list(current_best_params.values()) + [best_fit])
        print(f"Iter {t+1:03d} | Best MSE: {best_fit:.6f}")

    runtime = time.time() - start
    return current_best_params, best_fit, runtime, log

# =====================================================
# 7. EXECUTION
# =====================================================
# Extract bounds from configuration
lb = np.array([v[0] for v in MODEL["bounds"].values()])
ub = np.array([v[1] for v in MODEL["bounds"].values()])

# Create the objective function
objective_func = make_objective(MODEL["builder"], MODEL["bounds"])

# Run RPO
best_params, best_cv_mse, runtime, history = RPO(
    objective_func,
    lb,
    ub,
    N=CONFIG["population"],
    T=CONFIG["iterations"]
)

print("\n" + "="*40)
print("OPTIMIZATION COMPLETED")
print("="*40)
print(f"Best Params: {best_params}")
print(f"Best CV MSE: {best_cv_mse:.6f}")
print(f"Total Time : {runtime:.2f} seconds")

# =====================================================
# 8. FINAL MODEL TESTING
# =====================================================
final_model = ExtraTreesRegressor(
    **best_params, 
    random_state=CONFIG["random_state"], 
    n_jobs=-1
)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\n--- Final Test Results ---")
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.6f}")
print(f"Test R2 : {r2_score(y_test, y_pred):.6f}")