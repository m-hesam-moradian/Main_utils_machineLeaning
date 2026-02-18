import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
CONFIG = {
    "optimizer": "Lion",
    "population": 25,
    "iterations": 50, # RFC is faster, but Lion converges quickly
    "cv": 5,
    "random_state": 42
}

# =====================================================
# 2. MODEL DEFINITION (Random Forest Classifier)
# =====================================================
MODEL = {
    "name": "RFC",
    "builder": RandomForestClassifier,
    "bounds": {
        "n_estimators": (10, 300, int),
        "max_depth": (3, 30, int),
        "min_samples_split": (2, 20, int),
        "max_features": (0.1, 1.0, float)
    }
}

# Helper to map continuous optimizer values to model params
def decode_params(vec, bounds):
    params = {}
    for i, (key, (low, high, p_type)) in enumerate(bounds.items()):
        val = vec[i]
        if p_type == int:
            params[key] = int(round(val))
        else:
            params[key] = float(val)
    return params

# =====================================================
# 3. OBJECTIVE FUNCTION
# =====================================================
def make_objective(X, y):
    def objective(vec):
        params = decode_params(vec, MODEL["bounds"])
        model = MODEL["builder"](
            **params,
            random_state=CONFIG["random_state"],
            n_jobs=-1
        )
        # We maximize accuracy, but optimizers usually minimize.
        # So we return (1 - accuracy)
        score = cross_val_score(
            model, X, y,
            cv=CONFIG["cv"],
            scoring="accuracy",
            n_jobs=-1
        ).mean()
        return 1 - score 
    return objective

# =====================================================
# 4. LION OPTIMIZATION ALGORITHM
# =====================================================
def Lion(objective, lb, ub, N, T):
    """
    Lion Optimization Algorithm
    Simplified version focusing on the sign-based update and linear interpolation.
    """
    start = time.time()
    D = len(lb)
    
    # Initialize Population
    pop = lb + np.random.rand(N, D) * (ub - lb)
    fit = np.array([objective(pop[i]) for i in range(N)])
    
    # Track Best
    best_idx = np.argmin(fit)
    best_pos = pop[best_idx].copy()
    best_fit = fit[best_idx]
    
    # Lion specific parameters
    beta1 = 0.9
    beta2 = 0.99
    
    log = []
    
    for t in range(T):
        for i in range(N):
            # Lion Update Rule (Symbolic/Sign based)
            # Create a 'velocity' or update direction
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            
            # Interpolate between current and best
            c1 = beta1 * pop[i] + (1 - beta1) * best_pos
            # Add some sign-based momentum
            update_direction = np.sign(c1 - pop[i] + r1 * 0.1)
            
            # Apply movement
            new_pos = pop[i] + update_direction * (r2 * 0.05 * (ub - lb))
            new_pos = np.clip(new_pos, lb, ub)
            
            f = objective(new_pos)
            
            if f < fit[i]:
                fit[i] = f
                pop[i] = new_pos
                if f < best_fit:
                    best_pos, best_fit = new_pos.copy(), f
        
        best_decoded = decode_params(best_pos, MODEL["bounds"])
        log.append([t + 1, best_fit] + list(best_decoded.values()))
        print(f"Iter {t+1:03d} | Best Error: {best_fit:.4f} (Acc: {1-best_fit:.4f})")

    return best_decoded, 1 - best_fit, time.time() - start

# =====================================================
# 5. EXECUTION MOCKUP
# =====================================================
if __name__ == "__main__":
    # Placeholder for data loading
    # df = pd.read_excel(DATA_PATH, sheet_name="...")
    # X_train, X_test, y_train, y_test = ... 
    
    lb = np.array([v[0] for v in MODEL["bounds"].values()])
    ub = np.array([v[1] for v in MODEL["bounds"].values()])
    
    # Since I don't have your X/y, this is where you'd call it:
    # obj_func = make_objective(X_train, y_train)
    # best_params, best_acc, duration = Lion(obj_func, lb, ub, CONFIG["population"], CONFIG["iterations"])
    
    print("\nScript ready for RFC with Lion Optimizer.")