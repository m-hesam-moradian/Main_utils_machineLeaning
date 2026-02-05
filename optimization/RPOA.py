import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_GBR"

CONFIG = {
    "optimizer": "RPOA",
    "population": 25,
    "iterations": 200,
    "cv": 5,
    "random_state": 42
}

# =====================================================
# 2. LOAD DATA
# =====================================================
df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=CONFIG["random_state"]
)

# =====================================================
# 3. MODEL DEFINITION (GBR)
# =====================================================
MODEL = {
    "name": "Gradient Boosting Regression (GBR)",
    "builder": GradientBoostingRegressor,
    "bounds": {
        "n_estimators": (50, 300, int),
        "max_depth": (2, 6, int),
        "learning_rate": (0.01, 0.3, float),
        "subsample": (0.5, 1.0, float) # Added for better GBR tuning
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
        # Using default 'squared_error' for standard GBR
        model = model_builder(**params, random_state=CONFIG["random_state"])
        
        neg_mse = cross_val_score(
            model, X_train, y_train,
            cv=CONFIG["cv"],
            scoring="neg_mean_squared_error",
            n_jobs=-1
        ).mean()
        return np.sqrt(-neg_mse)
    return objective

# =====================================================
# 5. RPOA OPTIMIZER (Updated for Full Progress Printing)
# =====================================================
def RPOA(objective, lb, ub, N, T, cast):
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

    print(f"\n🚀 Starting {CONFIG['optimizer']} Optimization...")
    print("-" * 100)

    for t in range(T):
        # RPOA shrinking factor
        a = 2 * (1 - t / T) 
        
        for i in range(N):
            r1, r2 = np.random.rand(), np.random.rand()
            
            # Movement Logic
            if np.random.rand() < 0.5:
                D_best = np.abs(r1 * best_pos - pop[i])
                candidate = best_pos - a * D_best
            else:
                random_panda = pop[np.random.randint(N)]
                D_rand = np.abs(r2 * random_panda - pop[i])
                candidate = random_panda + a * D_rand
                
            candidate = np.clip(candidate, lb, ub)
            f = objective(candidate)
            
            if f < fit[i]:
                pop[i] = candidate
                fit[i] = f
                if f < best_fit:
                    best_pos, best_fit = candidate.copy(), f

        convergence.append(best_fit)
        
        # Decode the best parameters for printing
        best_decoded = decode_params(best_pos, MODEL["bounds"], cast)
        
        # Store for the log
        log.append([t + 1] + [best_decoded[k] for k in MODEL["bounds"]] + [best_fit])

        # --- THIS PRINTS EVERY ITERATION ---
        param_str = ", ".join([f"{k}: {v}" for k, v in best_decoded.items()])
        print(f"Iteration {t+1:03d}/{T} | Best RMSE: {best_fit:.6f} | Params: [{param_str}]")

    print("-" * 100)
    runtime = time.time() - start
    return decode_params(best_pos, MODEL["bounds"], cast), best_fit, convergence, runtime, log


# =====================================================
# 6. RUN OPTIMIZATION
# =====================================================
lb, ub, cast = bounds_to_arrays(MODEL["bounds"])
objective = make_objective(MODEL["builder"], MODEL["bounds"], cast)

best_params, best_rmse, convergence, runtime, log = RPOA(
    objective, lb, ub, CONFIG["population"], CONFIG["iterations"], cast
)

# =====================================================
# 7. FINAL MODEL TRAINING
# =====================================================
final_model = MODEL["builder"](**best_params, random_state=CONFIG["random_state"])
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