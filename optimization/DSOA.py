# --------------------------------------------------------------
#  DSOA + KNNC / ADAC  (Excel loading as you requested)
# --------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------- 1. Load data -----------------------------
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx"
sheet_name = "Data_after_KFold_QR"
target_column = "Remaining Useful Life "

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X_raw = df.drop(columns=[target_column]).values
y_raw = df[target_column].values

# Convert target to categorical classes
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Scale features (important for KNN)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Number of classes: {len(np.unique(y))}")
print(f"Samples: {len(y)}, Features: {X.shape[1]}")

# ------------------- 2. Choose model --------------------------
MODEL = "KNNC"          # <--- change to "ADAC" for Adaptive Gradient Boosting
# MODEL = "ADAC"

# --------------------------------------------------------------
#  DSOA implementation (class) - Domestic Sheep Optimization Algorithm
# --------------------------------------------------------------
class DSOA:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None, use_cv=True):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.use_cv = use_cv

    def _fitness(self, params):
        """Fitness = 1 - average CV accuracy"""
        if MODEL == "KNNC":
            n_neighbors = int(round(np.clip(params[0], 1, 50)))
            weights_idx = int(round(params[1])) % 2
            weights = ['uniform', 'distance'][weights_idx]
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                n_jobs=-1
            )
        else:  # ADAC (AdaBoost)
            n_estimators = int(round(np.clip(params[0], 10, 300)))
            learning_rate = np.clip(params[1], 0.01, 2.0)
            model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, val in kf.split(X, y):
            try:
                model.fit(X[tr], y[tr])
                pred = model.predict(X[val])
                scores.append(accuracy_score(y[val], pred))
            except:
                scores.append(0.0)
        return 1.0 - np.mean(scores)

    # ---------- main optimisation loop (DSOA) ----------
    def optimize(self):
        np.random.seed(42)
        pop = self.lb + np.random.rand(self.N, self.dim) * (self.ub - self.lb)
        fitness = np.array([self._fitness(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        gbest = pop[best_idx].copy()
        gbest_fit = fitness[best_idx]

        history = [gbest.copy()]

        start = time.time()
        for t in range(1, self.max_iter + 1):
            progress = t / self.max_iter

            for i in range(self.N):
                r = np.random.rand()

                if r < (1 - progress):   # Grazing / Exploration Phase
                    # Sheep grazing randomly or following other sheep
                    idx = np.random.randint(0, self.N)
                    direction = pop[idx] - pop[i]
                    step = np.random.uniform(0.5, 1.5, self.dim)
                    new_pos = pop[i] + step * direction + np.random.randn(self.dim) * 0.3

                else:                    # Herding / Exploitation Phase (following shepherd)
                    # Move toward best solution (shepherd / best ram)
                    direction = gbest - pop[i]
                    step_size = np.random.uniform(0.1, 0.6, self.dim) * (1 - progress)
                    new_pos = pop[i] + step_size * direction + np.random.normal(0, 0.1, self.dim)

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self._fitness(new_pos)

                # Greedy update (better grazing area)
                if new_fit < fitness[i]:
                    pop[i] = new_pos
                    fitness[i] = new_fit

                # Occasional "scared" jump (disturbance)
                if fitness[i] > np.mean(fitness) * 1.35 and np.random.rand() < 0.08:
                    pop[i] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                    fitness[i] = self._fitness(pop[i])

            # Update global best (best ram / shepherd position)
            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < gbest_fit:
                gbest = pop[new_best_idx].copy()
                gbest_fit = fitness[new_best_idx]

            history.append(gbest.copy())

        runtime = time.time() - start
        return gbest, gbest_fit, history, runtime


# --------------------------------------------------------------
#  3. Run optimisation
# --------------------------------------------------------------
if MODEL == "KNNC":
    dim = 2  # n_neighbors + weights
    lb = [1, 0]
    ub = [50, 1]
    print("Optimising K-Nearest Neighbors Classification (KNNC) with DSOA...")
else:  # ADAC
    dim = 2  # n_estimators + learning_rate
    lb = [10, 0.01]
    ub = [300, 2.0]
    print("Optimising Adaptive Gradient Boosting Classification (ADAC) with DSOA...")

dsoa = DSOA(N=30, max_iter=200, lb=lb, ub=ub, dim=dim, use_cv=True)
best_hp, best_loss, hp_history, runtime = dsoa.optimize()

# --------------------------------------------------------------
#  4. Show hyper-parameters for every iteration (sample)
# --------------------------------------------------------------
print("\n=== Hyper-parameters for every iteration (sample) ===")
for i in range(0, 10):
    print(f"Iter {i:3d}: {hp_history[i]}")
print(" ... ")
for i in range(-5, 0):
    print(f"Iter {len(hp_history)+i-1:3d}: {hp_history[i]}")

# --------------------------------------------------------------
#  5. Final model with best hyper-parameters
# --------------------------------------------------------------
if MODEL == "KNNC":
    n_neighbors = int(round(np.clip(best_hp[0], 1, 50)))
    weights_idx = int(round(best_hp[1])) % 2
    weights = ['uniform', 'distance'][weights_idx]
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1
    )
else:  # ADAC
    n_estimators = int(round(np.clip(best_hp[0], 10, 300)))
    learning_rate = np.clip(best_hp[1], 0.01, 2.0)
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

model.fit(X, y)
y_pred = model.predict(X)

# --------------------------------------------------------------
#  6. Metrics
# --------------------------------------------------------------
acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')

print("\n=== FINAL PERFORMANCE (on whole data) ===")
print(f"Run time           : {runtime:.2f} s")
print(f"Best Loss (1-Acc)  : {best_loss:.6f}")
print(f"Accuracy           : {acc:.6f}")
print(f"F1-Score (weighted): {f1:.6f}")

# --------------------------------------------------------------
#  7. Save hyper-parameter history
# --------------------------------------------------------------
hist_df = pd.DataFrame(hp_history, columns=[f"param_{i}" for i in range(dim)])
hist_df.to_excel("DSOA_hyperparameters_history.xlsx", index_label="iteration")
print("\nHyper-parameter history saved to 'DSOA_hyperparameters_history.xlsx'")