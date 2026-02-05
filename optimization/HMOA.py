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
    "optimizer": "HMOA",
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
# 3. MODEL DEFINITION (Huber Regression)
# =====================================================
MODEL = {
    "name": "Huber Regression (HR)",
    "builder": HuberRegressor,
    "bounds": {
        "epsilon": (1.0, 2.0, float),
        "alpha": (0.0001, 0.1, float),
        "max_iter": (100, 500, int)
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
            return 1e10 
    return objective

# =====================================================
# 5. HMOA OPTIMIZER (Human Memory Optimization Algorithm)
# =====================================================
def HMOA(objective, lb, ub, N, T, cast):
    start = time.time()
    D = len(lb)
    
    # Initialize "Memory Bank" (Population)
    pop = lb + np.random.rand(N, D) * (ub - lb)
    fit = np.array([objective(pop[i]) for i in range(N)])
    
    best_idx = np.argmin(fit)
    best_memory = pop[best_idx].copy()
    best_fit = fit[best_idx]
    
    convergence = []
    log = []

    print(f"\n🧠 Starting {CONFIG['optimizer']} Optimization...")
    print("-" * 100)

    for t in range(T):
        # Retention Rate (decreases over time to shift from exploration to exploitation)
        retention = 0.5 * (1 + np.cos(np.pi * t / T)) 
        
        for i in range(N):
            r = np.random.rand()
            
            if r < retention:
                # Sensory Memory Phase (Broad Exploration)
                # Learning from a random "experience" in the population
                rand_idx = np.random.randint(N)
                pop[i] = pop[i] + np.random.uniform(-1, 1, D) * (pop[rand_idx] - pop[i])
            else:
                # Long-Term Memory Phase (Refinement)
                # Focusing on the best memory found so far
                dist_to_best = best_memory - pop[i]
                pop[i] = pop[i] + np.random.rand() * dist_to_best
            
            # Boundary control
            pop[i] = np.clip(pop[i], lb, ub)
            
            # Re-evaluation
            f_new = objective(pop[i])
            if f_new < fit[i]:
                fit[i] = f_new
                if f_new < best_fit:
                    best_fit = f_new
                    best_memory = pop[i].copy()

        convergence.append(best_fit)
        best_decoded = decode_params(best_memory, MODEL["bounds"], cast)
        
        if (t + 1) % 10 == 0 or t == 0:
            param_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in best_decoded.items()])
            print(f"Iteration {t+1:03d}/{T} | Best RMSE: {best_fit:.6f} | Params: [{param_str}]")

    runtime = time.time() - start
    return best_decoded, best_fit, convergence, runtime

# =====================================================
# 6. RUN & EVALUATE
# =====================================================
lb, ub, cast = bounds_to_arrays(MODEL["bounds"])
objective = make_objective(MODEL["builder"], MODEL["bounds"], cast)

best_params, best_rmse, convergence, runtime = HMOA(
    objective, lb, ub, CONFIG["population"], CONFIG["iterations"], cast
)

# Final Training
final_model = MODEL["builder"](**best_params)
final_model.fit(X_train, y_train)
test_rmse = np.sqrt(mean_squared_error(y_test, final_model.predict(X_test)))

print("-" * 100)
print(f"✅ Final CV RMSE: {best_rmse:.6f}")
print(f"✅ Final Test RMSE: {test_rmse:.6f}")
print(f"✅ Optimized Parameters: {best_params}")