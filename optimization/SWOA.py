# --------------------------------------------------------------
#  SWOA + LDA / MLR  (Excel loading as you requested)
# --------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ------------------- 1. Load data -----------------------------
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx"
sheet_name = "Data_after_KFold_QR"   # Change if you have a different sheet for classification
target_column = "Remaining Useful Life "   # Change if target is categorical

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X = df.drop(columns=[target_column]).values
y_raw = df[target_column].values

# Convert target to categorical if needed (for LDA/MLR)
le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"Classes detected: {len(np.unique(y))}")

# ------------------- 2. Choose model --------------------------
MODEL = "LDA"          # <--- change to "MLR" for Multinomial Logistic Regression
# MODEL = "MLR"

# --------------------------------------------------------------
#  SWOA implementation (class) - Synergistic Swarm Optimization Algorithm
# --------------------------------------------------------------
class SWOA:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None, use_cv=True):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.use_cv = use_cv
        self.w = 0.9   # inertia weight (decreases over time)
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient

    def _fitness(self, params):
        """Fitness: 1 - average accuracy (or F1) over 5-fold CV"""
        if MODEL == "LDA":
            # For LDA, params can control solver, shrinkage, etc. (example: 2-3 params)
            solver_idx = int(round(params[0])) % 3
            solvers = ['svd', 'lsqr', 'eigen']
            shrinkage = params[1] if len(params) > 1 else 0.0
            model = LinearDiscriminantAnalysis(solver=solvers[solver_idx], shrinkage=shrinkage)
        else:  # MLR
            C = np.exp(params[0])  # Regularization strength
            model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                C=C,
                max_iter=500,
                random_state=42
            )

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, val in kf.split(X, y):
            X_tr, X_val = X[tr], X[val]
            y_tr, y_val = y[tr], y[val]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, pred))
        return 1.0 - np.mean(scores)  # Minimize (1 - accuracy)

    # ---------- main optimisation loop ----------
    def optimize(self):
        np.random.seed(42)
        pop = self.lb + np.random.rand(self.N, self.dim) * (self.ub - self.lb)
        velocity = np.zeros((self.N, self.dim))

        fitness = np.array([self._fitness(ind) for ind in pop])

        pbest = pop.copy()
        pbest_fit = fitness.copy()
        gbest_idx = np.argmin(fitness)
        gbest = pop[gbest_idx].copy()
        gbest_fit = fitness[gbest_idx]

        history = [gbest.copy()]

        start = time.time()
        for t in range(1, self.max_iter + 1):
            # Decrease inertia
            w = self.w * (1 - t / self.max_iter)

            for i in range(self.N):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Velocity update (synergistic swarm style)
                velocity[i] = (w * velocity[i] +
                               self.c1 * r1 * (pbest[i] - pop[i]) +
                               self.c2 * r2 * (gbest - pop[i]))

                # Position update
                pop[i] = pop[i] + velocity[i]
                pop[i] = np.clip(pop[i], self.lb, self.ub)

                # Evaluate
                new_fit = self._fitness(pop[i])

                # Greedy update
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
                    pbest[i] = pop[i].copy()

            # Update global best
            new_gbest_idx = np.argmin(fitness)
            if fitness[new_gbest_idx] < gbest_fit:
                gbest = pop[new_gbest_idx].copy()
                gbest_fit = fitness[new_gbest_idx]

            history.append(gbest.copy())

        runtime = time.time() - start
        return gbest, gbest_fit, history, runtime


# --------------------------------------------------------------
#  3. Run optimisation
# --------------------------------------------------------------
if MODEL == "LDA":
    dim = 2  # Example: solver index + shrinkage
    lb = [0, 0.0]
    ub = [2, 1.0]
    print("Optimising Linear Discriminant Analysis...")
else:  # MLR
    dim = 1  # Example: log(C) regularization
    lb = [-5.0]
    ub = [5.0]
    print("Optimising Multinomial Logistic Regression...")

swoa = SWOA(N=30, max_iter=200, lb=lb, ub=ub, dim=dim, use_cv=True)
best_hp, best_loss, hp_history, runtime = swoa.optimize()

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
if MODEL == "LDA":
    solver_idx = int(round(best_hp[0])) % 3
    solvers = ['svd', 'lsqr', 'eigen']
    shrinkage = best_hp[1] if len(best_hp) > 1 else 0.0
    model = LinearDiscriminantAnalysis(solver=solvers[solver_idx], shrinkage=shrinkage)
else:
    C = np.exp(best_hp[0])
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=C,
        max_iter=500,
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
print(f"Run time          : {runtime:.2f} s")
print(f"Best Loss (1-Acc) : {best_loss:.6f}")
print(f"Accuracy          : {acc:.6f}")
print(f"F1-Score (weighted): {f1:.6f}")

# --------------------------------------------------------------
#  7. Save hyper-parameter history
# --------------------------------------------------------------
hist_df = pd.DataFrame(hp_history, columns=[f"param_{i}" for i in range(dim)])
hist_df.to_excel("SWOA_hyperparameters_history.xlsx", index_label="iteration")
print("\nHyper-parameter history saved to 'SWOA_hyperparameters_history.xlsx'")