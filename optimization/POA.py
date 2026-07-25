# --------------------------------------------------------------
#  POA + Extra Trees Classifier (ETC)
# --------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------- 1. Load data -----------------------------
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx"
sheet_name = "Data_after_KFold_QR"
target_column = "Remaining Useful Life "

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X_raw = df.drop(columns=[target_column]).values
y_raw = df[target_column].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Classes: {len(np.unique(y))} | Samples: {len(y)} | Features: {X.shape[1]}")

# ------------------- 2. POA Class -----------------------------
class POA:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.alpha = 0.8

    def _fitness(self, params):
        n_estimators = int(round(np.clip(params[0], 50, 400)))
        max_depth = int(round(np.clip(params[1], 3, 30)))
        min_samples_split = int(round(np.clip(params[2], 2, 20)))
        min_samples_leaf = int(round(np.clip(params[3], 1, 10)))

        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
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
            exp_prob = self.alpha * (1 - t / self.max_iter)

            for i in range(self.N):
                r = np.random.rand()

                if r < exp_prob:
                    # Exploration: broad ingredient changes
                    idx = np.random.randint(0, self.N)
                    direction = pop[idx] - pop[i]
                    step = np.random.uniform(0.5, 1.5, self.dim)
                    new_pos = pop[i] + step * direction + np.random.randn(self.dim) * 0.2
                else:
                    # Exploitation: precise fine-tuning
                    delta = (gbest - pop[i]) * np.random.uniform(0.05, 0.3, self.dim)
                    new_pos = pop[i] + delta + np.random.normal(0, 0.06, self.dim)

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self._fitness(new_pos)

                if new_fit < fitness[i]:
                    pop[i] = new_pos
                    fitness[i] = new_fit

            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < gbest_fit:
                gbest = pop[new_best_idx].copy()
                gbest_fit = fitness[new_best_idx]

            history.append(gbest.copy())

        runtime = time.time() - start
        return gbest, gbest_fit, history, runtime

# ------------------- 3. Run -----------------------------------
dim = 4
lb = [50, 3, 2, 1]
ub = [400, 30, 20, 10]

print("Optimising Extra Trees Classifier with POA...")
poa = POA(N=30, max_iter=200, lb=lb, ub=ub, dim=dim)
best_hp, best_loss, hp_history, runtime = poa.optimize()

print("\n=== Hyper-parameters sample ===")
for i in range(0, 5):
    print(f"Iter {i:3d}: {hp_history[i]}")
print(" ... ")
for i in range(-3, 0):
    print(f"Iter {len(hp_history)+i-1:3d}: {hp_history[i]}")

# Final model
n_est = int(round(np.clip(best_hp[0], 50, 400)))
max_d = int(round(np.clip(best_hp[1], 3, 30)))
min_ss = int(round(np.clip(best_hp[2], 2, 20)))
min_sl = int(round(np.clip(best_hp[3], 1, 10)))

model = ExtraTreesClassifier(
    n_estimators=n_est,
    max_depth=max_d,
    min_samples_split=min_ss,
    min_samples_leaf=min_sl,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)
y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')

print("\n=== FINAL PERFORMANCE ===")
print(f"Run time           : {runtime:.2f} s")
print(f"Best Loss (1-Acc)  : {best_loss:.6f}")
print(f"Accuracy           : {acc:.6f}")
print(f"F1-Score (weighted): {f1:.6f}")
print(f"Best Params        : n_estimators={n_est}, max_depth={max_d}, min_samples_split={min_ss}, min_samples_leaf={min_sl}")

hist_df = pd.DataFrame(hp_history, columns=["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])
hist_df.to_excel("POA_ETC_hyperparameters_history.xlsx", index_label="iteration")
print("\nHistory saved to 'POA_ETC_hyperparameters_history.xlsx'")