import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx"
sheet_name = "Data_after_KFold_QR"   # Changed to your QR sheet

CONFIG = {
    "optimizer": "WEOA",          # You can change this later
    "population": 30,
    "iterations": 200,
    "cv": 5,
    "random_state": 42
}

# =====================================================
# 2. LOAD DATA
# =====================================================
df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
X = df.drop(columns=["Remaining Useful Life "]).values
y = df["Remaining Useful Life "].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=CONFIG["random_state"]
)

# =====================================================
# 3. MODEL DEFINITION (Quantile Regression)
# =====================================================
MODEL = {
    "name": "Quantile Regression (QR)",
    "builder": QuantileRegressor,
    "bounds": {
        "alpha": (0.0001, 1.0, float),      # Regularization strength
        "quantile": (0.5, 0.5, float)       # Fixed at 0.5 (median)
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
            return np.sqrt(-neg_mse)   # RMSE
        except:
            return 1e10
    return objective

# =====================================================
# 5. OPTIMIZER (You can replace with any previous optimizer)
# =====================================================
def WEOA(objective, lb, ub, N, T, cast):
    start = time.time()
    D = len(lb)
    
    pop = lb + np.random.rand(N, D) * (ub - lb)
    fit = np.array([objective(pop[i]) for i in range(N)])
    
    best_idx = np.argmin(fit)
    best_pos = pop[best_idx].copy()
    best_fit = fit[best_idx]
    
    convergence = []
    log = []

    print(f"\n🌿 Starting {CONFIG['optimizer']} Optimization for {MODEL['name']}...")
    print("-" * 100)

    for t in range(T):
        water_level = 1.0 - (t / T)
        
        for i in range(N):
            r = np.random.rand()

            if r < water_level:
                # Flooding / Exploration
                idx = np.random.randint(0, N)
                flow = (pop[idx] - pop[i]) * np.random.uniform(0.6, 1.4, D)
                candidate = pop[i] + flow + np.random.randn(D) * 0.25 * water_level
            else:
                # Drying / Exploitation
                growth = (best_pos - pop[i]) * np.random.uniform(0.1, 0.5, D)
                evaporation = np.random.normal(0, 0.1 * (1 - water_level), D)
                candidate = pop[i] + growth + evaporation

            candidate = np.clip(candidate, lb, ub)
            f_new = objective(candidate)

            if f_new < fit[i]:
                pop[i] = candidate
                fit[i] = f_new

            # Biodiversity reset
            if fit[i] > np.mean(fit) * 1.3 and np.random.rand() < 0.07:
                pop[i] = lb + np.random.rand(D) * (ub - lb)
                fit[i] = objective(pop[i])

        # Update global best
        new_best_idx = np.argmin(fit)
        if fit[new_best_idx] < best_fit:
            best_fit = fit[new_best_idx]
            best_pos = pop[new_best_idx].copy()

        convergence.append(best_fit)
        best_decoded = decode_params(best_pos, MODEL["bounds"], cast)
        log.append([t + 1] + [best_decoded[k] for k in MODEL["bounds"]] + [best_fit])

        if (t + 1) % 20 == 0 or t == 0:
            param_str = ", ".join([f"{k}: {v:.6f}" for k, v in best_decoded.items()])
            print(f"Iteration {t+1:03d}/{T} | Best CV-RMSE: {best_fit:.6f} | Params: [{param_str}]")

    print("-" * 100)
    runtime = time.time() - start
    return decode_params(best_pos, MODEL["bounds"], cast), best_fit, convergence, runtime, log

# =====================================================
# 6. RUN OPTIMIZATION
# =====================================================
lb, ub, cast = bounds_to_arrays(MODEL["bounds"])
objective = make_objective(MODEL["builder"], MODEL["bounds"], cast)

best_params, best_rmse, convergence, runtime, log = WEOA(
    objective, lb, ub, CONFIG["population"], CONFIG["iterations"], cast
)

# =====================================================
# 7. FINAL MODEL TRAINING & EVALUATION
# =====================================================
final_model = MODEL["builder"](**best_params)
final_model.fit(X_train, y_train)

y_pred_test = final_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

# =====================================================
# 8. OUTPUT
# =====================================================
summary_df = pd.DataFrame([{
    "Model": MODEL["name"],
    "Optimizer": CONFIG["optimizer"],
    "Best_CV_RMSE": best_rmse,
    "Test_RMSE": test_rmse,
    "Test_MAE": test_mae,
    "Test_R2": test_r2,
    "Runtime_sec": runtime
}])

print("\n✅ Final Summary:")
print(summary_df)
print("\n✅ Optimized Parameters:")
print(best_params)

# Save history
log_df = pd.DataFrame(log, columns=["Iteration"] + list(MODEL["bounds"].keys()) + ["RMSE"])
log_df.to_excel("WEOA_QR_Optimization_History.xlsx", index=False)
print("\nOptimization history saved to 'WEOA_QR_Optimization_History.xlsx'")