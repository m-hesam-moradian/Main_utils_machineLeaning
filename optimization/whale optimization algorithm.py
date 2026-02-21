import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# =====================================================
# 1. CONFIGURATION
# =====================================================
CONFIG = {
    "optimizer": "WOA",  # Whale Optimization Algorithm
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
        "C": (0.001, 1000.0, float),     
        "gamma": (0.0001, 10.0, float),   
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
        model = model_builder(
            **params,
            kernel='rbf',
            random_state=CONFIG["random_state"]
        )
        # Minimize negative accuracy
        score = cross_val_score(
            model, X_train, y_train,
            cv=CONFIG["cv"],
            scoring="accuracy",
            n_jobs=-1
        ).mean()
        return -score 
    return objective

# =====================================================
# 4. WHALE OPTIMIZATION ALGORITHM (WOA)
# =====================================================
def WOA(objective, lb, ub, N, T):
    """
    Whale Optimization Algorithm (WOA)
    Simulates the bubble-net hunting behavior of humpback whales.
    """
    start = time.time()
    D = len(lb)
    
    # Initialize Population (0 to 1 scaling)
    pop = np.random.uniform(0, 1, (N, D))
    
    # Initialize the leader (The Best Whale)
    leader_pos = np.zeros(D)
    leader_fit = np.inf

    log = []
    
    for t in range(T):
        # 1. Update Leader
        for i in range(N):
            # Boundary check
            pop[i] = np.clip(pop[i], 0, 1)
            fitness = objective(pop[i])
            
            if fitness < leader_fit:
                leader_fit = fitness
                leader_pos = pop[i].copy()

        # 2. Update Positions
        a = 2 - t * (2 / T)  # Linearly decreases from 2 to 0
        
        for i in range(N):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = np.random.rand()
            b = 1  # Constant for defining shape of logarithmic spiral

            if p < 0.5:
                if np.abs(A) < 1:
                    # Encircling Prey
                    D_leader = np.abs(C * leader_pos - pop[i])
                    pop[i] = leader_pos - A * D_leader
                else:
                    # Search for Prey (Exploration)
                    rand_whale = pop[np.random.randint(0, N)]
                    D_rand = np.abs(C * rand_whale - pop[i])
                    pop[i] = rand_whale - A * D_rand
            else:
                # Bubble-net Attacking (Spiral updating position)
                dist_to_leader = np.abs(leader_pos - pop[i])
                pop[i] = dist_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos

        best_decoded = decode_params(leader_pos, MODEL["bounds"])
        log.append([t + 1, -leader_fit])
        print(f"Iter {t+1:03d} | Best Accuracy: {-leader_fit:.4f}")

    return best_decoded, -leader_fit, time.time() - start, log