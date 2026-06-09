# --------------------------------------------------------------
#  CCO + LDA / MLR  (Excel loading as you requested)
# --------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ------------------- 1. Load data -----------------------------
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx"
sheet_name = "Data_after_KFold_QR"   # Change sheet name if needed for classification
target_column = "Remaining Useful Life "

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X = df.drop(columns=[target_column]).values
y_raw = df[target_column].values

# Convert target to categorical classes for LDA/MLR
le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"Number of classes detected: {len(np.unique(y))}")

# ------------------- 2. Choose model --------------------------
MODEL = "LDA"          # <--- change to "MLR" for Multinomial Logistic Regression
# MODEL = "MLR"

# --------------------------------------------------------------
#  CCO implementation (class) - Cuckoo Catfish Optimizer
# --------------------------------------------------------------
class CCO:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None, use_cv=True):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.use_cv = use_cv

    def _fitness(self, params):
        """Fitness = 1 - average CV accuracy"""
        if MODEL == "LDA":
            solver_idx = int(round(params[0])) % 3
            solvers = ['svd', 'lsqr', 'eigen']
            shrinkage = np.clip(params[1] if len(params) > 1 else 0.0, 0.0, 1.0)
            model = LinearDiscriminantAnalysis(solver=solvers[solver_idx], shrinkage=shrinkage)
        else:  # MLR
            C = np.exp(np.clip(params[0], -10, 10))  # Regularization strength
            model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                C=C,
                max_iter=1000,
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

    # ---------- main optimisation loop (CCO) ----------
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
            # Phase control
            progress = t / self.max_iter

            for i in range(self.N):
                r = np.random.rand()

                if progress < 0.4:          # Early: Exploration (Wraparound + Compressed Space)
                    # Wraparound search (multidimensional enveloping)
                    idx = np.random.randint(0, self.N)
                    new_pos = pop[i] + np.random.uniform(-1, 2, self.dim) * (pop[idx] - pop[i])

                elif progress < 0.7:        # Transition phase
                    # Divide population behavior
                    if r < 0.5:
                        new_pos = gbest + np.random.randn(self.dim) * 0.3 * (self.ub - self.lb)  # Near best
                    else:
                        new_pos = pop[i] + np.random.uniform(-1.5, 1.5, self.dim) * (pop[i] - gbest)  # Away from best

                else:                       # Exploitation: Chaotic Predation & Parasitism
                    # Chaotic perturbation around best
                    chaotic = np.random.uniform(0, 1) * np.sin(t) * (gbest - pop[i])
                    new_pos = gbest + chaotic + np.random.randn(self.dim) * 0.1

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self._fitness(new_pos)

                # Greedy selection (predation success)
                if new_fit < fitness[i]:
                    pop[i] = new_pos
                    fitness[i] = new_fit

                # Death & rebirth (if poor performance)
                if fitness[i] > np.mean(fitness) * 1.5 and np.random.rand() < 0.1:
                    pop[i] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                    fitness[i] = self._fitness(pop[i])

            # Update global best
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
if MODEL == "LDA":
    dim = 2
    lb = [0.0, 0.0]
    ub = [2.0, 1.0]
    print("Optimising Linear Discriminant Analysis with CCO...")
else:  # MLR
    dim = 1
    lb = [-5.0]
    ub = [5.0]
    print("Optimising Multinomial Logistic Regression with CCO...")

cco = CCO(N=30, max_iter=200, lb=lb, ub=ub, dim=dim, use_cv=True)
best_hp, best_loss, hp_history, runtime = cco.optimize()

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
    shrinkage = np.clip(best_hp[1] if len(best_hp) > 1 else 0.0, 0.0, 1.0)
    model = LinearDiscriminantAnalysis(solver=solvers[solver_idx], shrinkage=shrinkage)
else:
    C = np.exp(np.clip(best_hp[0], -10, 10))
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=C,
        max_iter=1000,
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
hist_df.to_excel("CCO_hyperparameters_history.xlsx", index_label="iteration")
print("\nHyper-parameter history saved to 'CCO_hyperparameters_history.xlsx'")