import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error  # Used in RAE calc

excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx"
sheet_name = "Data_after_KFold_QR"
target_column = "Remaining Useful Life "

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X = df.drop(columns=[target_column])
y = df[target_column]

N = 30  # population size
D = 3   # dimensions (n_estimators, max_depth, min_samples_split)
max_iter = 200
lb = np.array([50, 5, 2])
ub = np.array([300, 30, 20])

# Fitness function: Average RAE over 5-fold CV
def fitness(individual, X, y):
    params = np.round(individual).astype(int)
    n_est, max_d, min_ss = params
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    raes = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = ExtraTreesRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_ss,
            random_state=42
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        abs_err = np.abs(y_val - pred)
        abs_dev = np.abs(y_val - np.mean(y_tr))
        rae = np.sum(abs_err) / np.sum(abs_dev) if np.sum(abs_dev) > 0 else 0
        raes.append(rae)
    return np.mean(raes)

# Initialize population
np.random.seed(42)  # For reproducibility
X_pop = lb + np.random.rand(N, D) * (ub - lb)
fitness_pop = np.array([fitness(ind, X, y) for ind in X_pop])
pbest = X_pop.copy()
fitness_pbest = fitness_pop.copy()
gbest_idx = np.argmin(fitness_pop)
gbest = X_pop[gbest_idx].copy()
fitness_gbest = fitness_pop[gbest_idx]

T_max = 40
F_max = 3

for t in range(1, max_iter + 1):
    T = (t / max_iter) * T_max
    for i in range(N):
        if T > 30:
            # Competition phase
            C = (pbest[i] + gbest) / 2
            r = np.random.rand()
            if r > 0.5:
                X_pop[i] = C
            else:
                rand_idx = np.random.randint(0, N)
                X_pop[i] = X_pop[rand_idx]
        else:
            # Foraging phase
            F_loc = gbest
            fi = fitness_pop[i]
            ff = fitness_gbest
            diff = np.abs(fitness_pop - ff)
            max_diff = np.max(diff) if np.max(diff) > 0 else 1e-6
            S = F_max * np.abs(fi - ff) / max_diff
            r = np.random.rand()
            if S > 2:
                if np.random.rand() < 0.5:
                    trig = np.sin(np.pi * r)
                else:
                    trig = np.cos(np.pi * r)
                X_pop[i] = F_loc + F_loc * trig
            else:
                X_pop[i] = F_loc + r * (F_loc - X_pop[i])
        # Clip to bounds
        X_pop[i] = np.clip(X_pop[i], lb, ub)
    # Update fitness
    fitness_pop = np.array([fitness(ind, X, y) for ind in X_pop])
    # Update pbest
    better = fitness_pop < fitness_pbest
    pbest[better] = X_pop[better]
    fitness_pbest[better] = fitness_pop[better]
    # Update gbest
    new_gbest_idx = np.argmin(fitness_pop)
    if fitness_pop[new_gbest_idx] < fitness_gbest:
        gbest = X_pop[new_gbest_idx].copy()
        fitness_gbest = fitness_pop[new_gbest_idx]

# Round final to integers
optimal_params = np.round(gbest).astype(int)
print("Optimal hyperparameters: n_estimators={}, max_depth={}, min_samples_split={}".format(*optimal_params))
print("Final average RAE (from CV):", fitness_gbest)