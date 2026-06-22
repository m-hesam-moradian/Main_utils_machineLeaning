# --------------------------------------------------------------
#  WEOA + LR / LSSVC  (Excel loading as you requested)
# --------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------- 1. Load data -----------------------------
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx"
sheet_name = "Data_after_KFold_QR"
target_column = "Remaining Useful Life "

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X_raw = df.drop(columns=[target_column]).values
y_raw = df[target_column].values

# Convert target to categorical
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Scale features (very important for LR and LSSVC)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Number of classes: {len(np.unique(y))}")
print(f"Samples: {len(y)}, Features: {X.shape[1]}")

# ------------------- 2. Choose model --------------------------
MODEL = "LR"          # <--- change to "LSSVC" for Least Squares Support Vector Classification
# MODEL = "LSSVC"

# --------------------------------------------------------------
#  WEOA implementation (class) - Wetland Ecosystem Optimization Algorithm
# --------------------------------------------------------------
class WEOA:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None, use_cv=True):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.use_cv = use_cv

    def _fitness(self, params):
        """Fitness = 1 - average CV accuracy"""
        if MODEL == "LR":
            C = np.exp(np.clip(params[0], -10, 10))
            l1_ratio = np.clip(params[1] if self.dim > 1 else 0.0, 0.0, 1.0)
            model = LogisticRegression(
                multi_class='multinomial',
                solver='saga',
                penalty='elasticnet',
                C=C,
                l1_ratio=l1_ratio,
                max_iter=1000,
                random_state=42
            )
        else:  # LSSVC approximation
            C = np.exp(np.clip(params[0], -5, 10))
            gamma = np.exp(np.clip(params[1] if self.dim > 1 else 0.0, -5, 5))
            model = SVC(
                kernel='rbf',
                C=C,
                gamma=gamma,
                probability=True,
                random_state=42,
                max_iter=1000
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

    # ---------- main optimisation loop (WEOA) ----------
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
            water_level = 1.0 - progress   # Simulates drying/flooding cycle

            for i in range(self.N):
                r = np.random.rand()

                if r < water_level:  
                    # === Flooding / Nutrient Flow Phase (Exploration) ===
                    idx = np.random.randint(0, self.N)
                    flow = (pop[idx] - pop[i]) * np.random.uniform(0.6, 1.4, self.dim)
                    new_pos = pop[i] + flow + np.random.randn(self.dim) * 0.25 * water_level

                else:
                    # === Drying / Plant Growth & Rooting Phase (Exploitation) ===
                    # Growth toward best solution (optimal wetland conditions)
                    growth = (gbest - pop[i]) * np.random.uniform(0.1, 0.5, self.dim)
                    evaporation = np.random.normal(0, 0.1 * (1 - water_level), self.dim)
                    new_pos = pop[i] + growth + evaporation

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self._fitness(new_pos)

                # Survival of the fittest (greedy selection)
                if new_fit < fitness[i]:
                    pop[i] = new_pos
                    fitness[i] = new_fit

                # Biodiversity reset (random species migration / disturbance)
                if fitness[i] > np.mean(fitness) * 1.3 and np.random.rand() < 0.07:
                    pop[i] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                    fitness[i] = self._fitness(pop[i])

            # Update global best (healthiest ecosystem)
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
if MODEL == "LR":
    dim = 2  # log(C) + l1_ratio
    lb = [-5.0, 0.0]
    ub = [5.0, 1.0]
    print("Optimising Logistic Regression (LR) with WEOA...")
else:  # LSSVC
    dim = 2  # log(C) + log(gamma)
    lb = [-5.0, -5.0]
    ub = [10.0, 5.0]
    print("Optimising Least Squares Support Vector Classification (LSSVC) with WEOA...")

weoa = WEOA(N=30, max_iter=200, lb=lb, ub=ub, dim=dim, use_cv=True)
best_hp, best_loss, hp_history, runtime = weoa.optimize()

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
if MODEL == "LR":
    C = np.exp(np.clip(best_hp[0], -10, 10))
    l1_ratio = np.clip(best_hp[1], 0.0, 1.0)
    model = LogisticRegression(
        multi_class='multinomial',
        solver='saga',
        penalty='elasticnet',
        C=C,
        l1_ratio=l1_ratio,
        max_iter=1000,
        random_state=42
    )
else:  # LSSVC
    C = np.exp(np.clip(best_hp[0], -10, 10))
    gamma = np.exp(np.clip(best_hp[1], -10, 10))
    model = SVC(
        kernel='rbf',
        C=C,
        gamma=gamma,
        probability=True,
        random_state=42,
        max_iter=1000
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
hist_df.to_excel("WEOA_hyperparameters_history.xlsx", index_label="iteration")
print("\nHyper-parameter history saved to 'WEOA_hyperparameters_history.xlsx'")