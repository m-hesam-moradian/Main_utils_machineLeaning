import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_GBR"

CONFIG = {
    "optimizer": "MAOA",
    "population": 25,
    "iterations": 200,
    "cv": 5,
    "random_state": 42
}

# =====================================================
# 2. LOAD DATA
# =====================================================
# Assuming the file exists at the path provided
df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=CONFIG["random_state"]
)

# =====================================================
# 3. MODEL DEFINITION (Huber Regression)
# =====================================================
MODEL = {
    "name": "Huber Regression (HR)",
    "builder": HuberRegressor,
    "bounds": {
        "epsilon": (1.0, 2.0, float),   # Threshold for considering points outliers
        "alpha": (0.0001, 0.1, float),  # L2 regularization parameter
        "max_iter": (100, 500, int)     # Maximum iterations for solver
    }
}

# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================
def bounds_to_arrays(bounds):
    lb, ub, cast = [], [], []
    for v in bounds.values():
        lb.append(v[0])
        ub.append(v[1])
        cast.append(v[2])
    return np.array(lb), np.array(ub), cast

def decode_params(vec, bounds, cast):
    decoded = {}
    for i, k in enumerate(bounds.keys()):
        decoded[k] = cast[i](vec[i])
    return decoded

def make_objective(model_builder, bounds, cast):
    def objective(vec):
        params = decode_params(vec, bounds, cast)
        model = model_builder(**params)
        
        try:
            neg_mse = cross_val_score(
                model, X_train, y_train,
                cv=CONFIG["cv"],
                scoring="neg_mean_squared_error",
                n_jobs=-1
            ).mean()
            return np.sqrt(-neg_mse)
        except:
            return 1e10 # Return large error if solver fails to converge
    return objective

# =====================================================
# 5. MAOA OPTIMIZER (Makeup Artist Optimization Algorithm)
# =====================================================
def MAOA(objective, lb, ub, N, T, cast):
    start = time.time()
    D = len(lb)
    
    # Initialize Population
    pop = lb + np.random.rand(N, D) * (ub - lb)
    fit = np.array([objective(pop[i]) for i in range(N)])
    
    best_idx = np.argmin(fit)
    best_pos = pop[best_idx].copy()
    best_fit = fit[best_idx]
    
    convergence = []
    log = []

    print(f"\n💄 Starting {CONFIG['optimizer']} Optimization (Huber Regression)...")
    print("-" * 100)

    for t in range(T):
        for i in range(N):
            # Phase 1: Makeup Artist (Exploration)
            # Modeling the artist's ability to find the best look
            r1 = np.random.rand()
            random_artist = pop[np.random.randint(N)]
            
            if r1 < 0.5:
                # Update based on best artist (best_pos)
                step = np.random.rand(D) * (best_pos - pop[i])
                candidate = pop[i] + step
            else:
                # Update based on peer influence
                step = np.random.rand(D) * (random_artist - pop[i])
                candidate = pop[i] + step

            # Boundary Check & Evaluation
            candidate = np.clip(candidate, lb, ub)
            f_new = objective(candidate)

            if f_new < fit[i]:
                pop[i] = candidate
                fit[i] = f_new

            # Phase 2: Client Satisfaction (Exploitation)
            # Local refinement around the best solution
            L = 0.2 * (1 - t/T) # Shrinking local search range
            candidate = pop[i] + (np.random.uniform(-1, 1, D) * L * (ub - lb))
            
            candidate = np.clip(candidate, lb, ub)
            f_new = objective(candidate)

            if f_new < fit[i]:
                pop[i] = candidate
                fit[i] = f_new
            
            # Global Best Update
            if fit[i] < best_fit:
                best_fit = fit[i]
                best_pos = pop[i].copy()

        convergence.append(best_fit)
        best_decoded = decode_params(best_pos, MODEL["bounds"], cast)
        log.append([t + 1] + [best_decoded[k] for k in MODEL["bounds"]] + [best_fit])

        if (t + 1) % 10 == 0 or t == 0:
            param_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in best_decoded.items()])
            print(f"Iteration {t+1:03d}/{T} | Best RMSE: {best_fit:.6f} | Params: [{param_str}]")

    print("-" * 100)
    runtime = time.time() - start
    return decode_params(best_pos, MODEL["bounds"], cast), best_fit, convergence, runtime, log

# =====================================================
# 6. RUN OPTIMIZATION
# =====================================================
lb, ub, cast = bounds_to_arrays(MODEL["bounds"])
objective = make_objective(MODEL["builder"], MODEL["bounds"], cast)

best_params, best_rmse, convergence, runtime, log = MAOA(
    objective, lb, ub, CONFIG["population"], CONFIG["iterations"], cast
)

# =====================================================
# 7. FINAL MODEL TRAINING
# =====================================================
final_model = MODEL["builder"](**best_params)
final_model.fit(X_train, y_train)

y_pred_test = final_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# =====================================================
# 8. OUTPUT DATASETS
# =====================================================
summary_df = pd.DataFrame([{
    "Model": MODEL["name"],
    "Optimizer": CONFIG["optimizer"],
    "Best_CV_RMSE": best_rmse,
    "Test_RMSE": test_rmse,
    "Runtime_sec": runtime
}])

print("\n✅ Final Summary:")
print(summary_df)
print("\n✅ Optimized Parameters:")
print(best_params)