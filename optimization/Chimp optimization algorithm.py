import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# =====================================================
# 1. CONFIGURATION
# =====================================================
CONFIG = {
    "optimizer": "ChO",  # Chimp Optimization Algorithm
    "population": 25,
    "iterations": 50,    
    "cv": 5,
    "random_state": 42
}

# =====================================================
# 2. MODEL DEFINITION (SVC)
# =====================================================
MODEL = {
    "name": "SVC",
    "builder": SVC,
    "bounds": {
        "C": (0.001, 1000.0, float),     # Regularization parameter
        "gamma": (0.0001, 10.0, float),   # Kernel coefficient
    }
}

def decode_params(vec, bounds):
    params = {}
    for i, (key, (low, high, cast)) in enumerate(bounds.items()):
        val = low + vec[i] * (high - low)
        params[key] = cast(val)
    return params

def make_objective(model_builder, bounds):
    def objective(vec):
        params = decode_params(vec, bounds)
        # We use SVC for classification
        model = model_builder(
            **params,
            kernel='rbf',
            random_state=CONFIG["random_state"]
        )
        # Optimize for Accuracy (Maximize, so we return negative for the optimizer)
        score = cross_val_score(
            model, X_train, y_train,
            cv=CONFIG["cv"],
            scoring="accuracy",
            n_jobs=-1
        ).mean()
        return -score 
    return objective

# =====================================================
# 4. CHIMP OPTIMIZATION ALGORITHM (ChO)
# =====================================================
def ChO(objective, lb, ub, N, T):
    """
    Chimp Optimization Algorithm (ChO)
    Mimics the social hierarchy and hunting behavior of chimps.
    """
    start = time.time()
    D = len(lb)
    
    # Initialize Population (Chimps)
    pop = np.random.uniform(0, 1, (N, D))
    
    # Initialize the four leaders (Attacker, Barrier, Chaser, Driver)
    best_pos = np.zeros((4, D)) 
    best_fit = np.full(4, np.inf)

    log = []
    
    for t in range(T):
        # 1. Evaluate and Update Leaders
        for i in range(N):
            fitness = objective(pop[i])
            
            # Update the 4 leaders based on fitness
            if fitness < best_fit[0]: # Attacker
                best_fit[0], best_pos[0] = fitness, pop[i].copy()
            elif fitness < best_fit[1] and fitness > best_fit[0]: # Barrier
                best_fit[1], best_pos[1] = fitness, pop[i].copy()
            elif fitness < best_fit[2] and fitness > best_fit[1]: # Chaser
                best_fit[2], best_pos[2] = fitness, pop[i].copy()
            elif fitness < best_fit[3] and fitness > best_fit[2]: # Driver
                best_fit[3], best_pos[3] = fitness, pop[i].copy()

        # 2. Update Positions
        f = 2 - t * (2 / T) # Non-linearly decreased from 2 to 0
        
        for i in range(N):
            new_pos = np.zeros(D)
            for j in range(4): # Influence from the 4 leaders
                C = 2 * np.random.rand()
                m = np.random.rand(D) # chaotic map factor
                
                # Hunting logic
                D_chimp = np.abs(C * best_pos[j] - m * pop[i])
                A = f * (2 * np.random.rand() - 1)
                
                new_pos += (best_pos[j] - A * D_chimp)
            
            pop[i] = np.clip(new_pos / 4, 0, 1)

        best_decoded = decode_params(best_pos[0], MODEL["bounds"])
        log.append([t + 1, -best_fit[0]])
        print(f"Iter {t+1:03d} | Best Accuracy: {-best_fit[0]:.4f}")

    return best_decoded, -best_fit[0], time.time() - start, log