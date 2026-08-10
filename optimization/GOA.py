# --------------------------------------------------------------
#  GOA + ENR / SVR
# --------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ------------------- 1. Load data -----------------------------
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx"
sheet_name = "Data_after_KFold_QR"
target_column = "Remaining Useful Life "

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X_raw = df.drop(columns=[target_column]).values
y = df[target_column].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Samples: {len(y)} | Features: {X.shape[1]}")

# ------------------- 2. Choose model --------------------------
MODEL = "ENR"          # <--- change to "SVR" for Support Vector Regression
# MODEL = "SVR"

# --------------------------------------------------------------
#  GOA implementation (Gazelle Optimization Algorithm)
# --------------------------------------------------------------
class GOA:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim

    def _fitness(self, params):
        if MODEL == "ENR":
            alpha = np.exp(np.clip(params[0], -8, 2))          # regularization strength
            l1_ratio = np.clip(params[1], 0.0, 1.0)           # mix of L1/L2
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=2000,
                random_state=42
            )
        else:  # SVR
            C = np.exp(np.clip(params[0], -2, 8))
            epsilon = np.clip(params[1], 0.001, 1.0)
            gamma = np.exp(np.clip(params[2] if self.dim > 2 else 0.0, -5, 3))
            model = SVR(
                kernel='rbf',
                C=C,
                epsilon=epsilon,
                gamma=gamma
            )

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses = []
        for tr, val in kf.split(X):
            try:
                model.fit(X[tr], y[tr])
                pred = model.predict(X[val])
                rmse = np.sqrt(mean_squared_error(y[val], pred))
                rmses.append(rmse)
            except:
                rmses.append(1e6)
        return np.mean(rmses)

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
            # Predator detection probability decreases over time
            predator_prob = 0.5 * (1 - progress)

            for i in range(self.N):
                r = np.random.rand()

                if r < predator_prob:
                    # Escape / Exploration (Levy-like jump + Brownian)
                    levy = np.random.standard_cauchy(self.dim) * 0.1 * (1 - progress)
                    brownian = np.random.randn(self.dim) * 0.2
                    new_pos = pop[i] + levy + brownian * (self.ub - self.lb) * 0.15
                else:
                    # Grazing / Exploitation toward best
                    direction = gbest - pop[i]
                    step = np.random.uniform(0.1, 0.6, self.dim) * (1 - progress)
                    new_pos = pop[i] + step * direction + np.random.normal(0, 0.05, self.dim)

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self._fitness(new_pos)

                if new_fit < fitness[i]:
                    pop[i] = new_pos
                    fitness[i] = new_fit

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
if MODEL == "ENR":
    dim = 2
    lb = [-8.0, 0.0]
    ub = [2.0, 1.0]
    print("Optimising ElasticNet Regression (ENR) with GOA...")
else:  # SVR
    dim = 3
    lb = [-2.0, 0.001, -5.0]
    ub = [8.0, 1.0, 3.0]
    print("Optimising Support Vector Regression (SVR) with GOA...")

goa = GOA(N=30, max_iter=200, lb=lb, ub=ub, dim=dim)
best_hp, best_rmse, hp_history, runtime = goa.optimize()

# --------------------------------------------------------------
#  4. Show sample of hyper-parameters
# --------------------------------------------------------------
print("\n=== Hyper-parameters sample ===")
for i in range(0, 5):
    print(f"Iter {i:3d}: {hp_history[i]}")
print(" ... ")
for i in range(-3, 0):
    print(f"Iter {len(hp_history)+i-1:3d}: {hp_history[i]}")

# --------------------------------------------------------------
#  5. Final model
# --------------------------------------------------------------
if MODEL == "ENR":
    alpha = np.exp(np.clip(best_hp[0], -8, 2))
    l1_ratio = np.clip(best_hp[1], 0.0, 1.0)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
    print(f"\nBest Params → alpha={alpha:.6f}, l1_ratio={l1_ratio:.4f}")
else:
    C = np.exp(np.clip(best_hp[0], -2, 8))
    epsilon = np.clip(best_hp[1], 0.001, 1.0)
    gamma = np.exp(np.clip(best_hp[2], -5, 3))
    model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
    print(f"\nBest Params → C={C:.4f}, epsilon={epsilon:.4f}, gamma={gamma:.6f}")

model.fit(X, y)
y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n=== FINAL PERFORMANCE (on whole data) ===")
print(f"Run time   : {runtime:.2f} s")
print(f"Best CV RMSE: {best_rmse:.6f}")
print(f"RMSE       : {rmse:.6f}")
print(f"MAE        : {mae:.6f}")
print(f"R²         : {r2:.6f}")

# --------------------------------------------------------------
#  6. Save history
# --------------------------------------------------------------
cols = [f"param_{i}" for i in range(dim)]
hist_df = pd.DataFrame(hp_history, columns=cols)
hist_df.to_excel(f"GOA_{MODEL}_hyperparameters_history.xlsx", index_label="iteration")
print(f"\nHistory saved to 'GOA_{MODEL}_hyperparameters_history.xlsx'")