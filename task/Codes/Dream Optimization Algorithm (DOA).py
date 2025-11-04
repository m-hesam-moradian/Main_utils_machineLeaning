# --------------------------------------------------------------
#  DOA + QR / ETR  (Excel loading as you requested)
# --------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------- 1. Load data -----------------------------
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.23-Data.xlsx"
sheet_name = "Data_after_KFold_QR"
target_column = "Remaining Useful Life "

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X = df.drop(columns=[target_column]).values          # numpy array
y = df[target_column].values

# ------------------- 2. Choose model --------------------------
MODEL = "QR"          # <--- change to "ETR" if you want Extra-Trees
# MODEL = "ETR"

# --------------------------------------------------------------
#  DOA implementation (class)
# --------------------------------------------------------------
class DOA:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None, use_cv=True):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.use_cv = use_cv
        self.p_mem = 0.1  # Probability for memory reset
        self.p_f = 0.1    # Probability for forgetting and supplementation
        self.p_recomb = 0.5  # Probability for recombination
        self.phase_switch = 5  # Switch phase every 5 iterations
        self.diversity_threshold = 1e-3  # Diversity threshold for phase decision

    def _rae(self, y_true, y_pred, y_train_mean):
        ae = np.abs(y_true - y_pred).sum()
        ad = np.abs(y_true - y_train_mean).sum()
        return ae / ad if ad > 0 else 0.0

    # ---------- fitness for QR (8 coefficients) ----------
    def _qr_fitness(self, beta):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        raes = []
        for tr, val in kf.split(X):
            X_tr, X_val = X[tr], X[val]
            y_tr, y_val = y[tr], y[val]

            # add intercept column
            X_tr_int = np.column_stack([np.ones(len(X_tr)), X_tr])
            X_val_int = np.column_stack([np.ones(len(X_val)), X_val])

            pred = X_val_int @ beta
            raes.append(self._rae(y_val, pred, y_tr.mean()))
        return np.mean(raes)

    # ---------- fitness for ETR (3 hyper-params) ----------
    def _etr_fitness(self, hp):
        n_est, max_d, min_ss = int(round(hp[0])), int(round(hp[1])), int(round(hp[2]))
        n_est = max(n_est, 1)
        max_d = max(max_d, 1)
        min_ss = max(min_ss, 2)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        raes = []
        for tr, val in kf.split(X):
            X_tr, X_val = X[tr], X[val]
            y_tr, y_val = y[tr], y[val]

            model = ExtraTreesRegressor(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_split=min_ss,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            raes.append(self._rae(y_val, pred, y_tr.mean()))
        return np.mean(raes)

    # ---------- dream recombination ----------
    def _dream_recombine(self, parent1, parent2):
        mask = np.random.rand(self.dim) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    # ---------- dream perturbation ----------
    def _dream_perturbation(self, x):
        sigma = 0.1 * (self.ub - self.lb)
        return np.random.normal(0, sigma, self.dim)

    # ---------- main optimisation loop ----------
    def optimize(self):
        np.random.seed(42)
        pop = self.lb + np.random.rand(self.N, self.dim) * (self.ub - self.lb)
        fitness = np.array([self._fitness(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)
        gbest = pop[best_idx].copy()
        gbest_fit = fitness[best_idx]
        gworst = pop[worst_idx].copy()

        history = [gbest.copy()]  # iteration 0 (initial)

        start = time.time()
        for t in range(1, self.max_iter + 1):
            # Compute diversity
            diversity = np.mean(np.std(pop, axis=0))

            # Decide phase
            if (t % self.phase_switch == 0) or (diversity < self.diversity_threshold):
                phase = 'exploration'
            else:
                phase = 'exploitation'

            # Memory reset
            if np.random.rand() < self.p_mem:
                for i in range(self.N):
                    if np.random.rand() < 0.5:
                        pop[i] = gbest.copy()

            # Forgetting and supplementation
            if np.random.rand() < self.p_f:
                num_forget = self.N // 2
                forget_idx = np.random.choice(self.N, num_forget, replace=False)
                pop[forget_idx] = self.lb + np.random.rand(num_forget, self.dim) * (self.ub - self.lb)

            # Dream-logic recombination
            for i in range(self.N):
                if np.random.rand() < self.p_recomb:
                    idx1, idx2 = np.random.choice(self.N, 2, replace=False)
                    parent1 = pop[idx1]
                    parent2 = pop[idx2]
                    pop[i] = self._dream_recombine(parent1, parent2)

            # Update positions based on phase
            for i in range(self.N):
                if phase == 'exploration':
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    r3 = np.random.rand()
                    perturbation = self._dream_perturbation(pop[i])
                    new_pos = pop[i] + r1 * (gbest - pop[i]) + r2 * (gworst - pop[i]) + r3 * perturbation
                else:  # exploitation
                    r = np.random.rand()
                    mutation = np.random.normal(0, 0.01 * (self.ub - self.lb), self.dim)
                    new_pos = gbest + r * (pop[i] - gbest) + mutation

                new_pos = np.clip(new_pos, self.lb, self.ub)
                pop[i] = new_pos

            # Update fitness
            fitness = np.array([self._fitness(ind) for ind in pop])

            # Update best and worst
            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < gbest_fit:
                gbest = pop[new_best_idx].copy()
                gbest_fit = fitness[new_best_idx]

            new_worst_idx = np.argmax(fitness)
            gworst = pop[new_worst_idx].copy()

            history.append(gbest.copy())

        runtime = time.time() - start
        return gbest, gbest_fit, history, runtime

    # ---------- select correct fitness ----------
    def _fitness(self, ind):
        if MODEL == "QR":
            return self._qr_fitness(ind)
        else:  # ETR
            return self._etr_fitness(ind)


# --------------------------------------------------------------
#  3. Run optimisation
# --------------------------------------------------------------
if MODEL == "QR":
    # 8 coefficients (intercept + 7 features)
    dim = 8
    lb = [-10.0] * dim
    ub = [ 10.0] * dim
    print("Optimising Quantile Regression (8 β coefficients)…")
else:   # ETR
    dim = 3
    lb = [50, 5, 2]                     # n_estimators, max_depth, min_samples_split
    ub = [300, 30, 20]
    print("Optimising Extra-Trees (n_estimators, max_depth, min_samples_split)…")

doa = DOA(N=30, max_iter=200, lb=lb, ub=ub, dim=dim, use_cv=True)
best_hp, best_rae, hp_history, runtime = doa.optimize()

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
#  5. Final model with the *best* hyper-parameters
# --------------------------------------------------------------
if MODEL == "QR":
    # build final linear predictor
    intercept = best_hp[0]
    coefs     = best_hp[1:]
    X_int = np.column_stack([np.ones(len(X)), X])
    y_pred = X_int @ best_hp
else:
    n_est = int(round(best_hp[0]))
    max_d = int(round(best_hp[1]))
    min_ss = int(round(best_hp[2]))
    etr = ExtraTreesRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        min_samples_split=min_ss,
        random_state=42,
        n_jobs=-1
    )
    etr.fit(X, y)
    y_pred = etr.predict(X)

# --------------------------------------------------------------
#  6. Metrics (R2, RMSE, RAE, U95, MARD)
# --------------------------------------------------------------
def calc_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # RAE
    mean_y = y_true.mean()
    rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - mean_y))

    # U95  (95-th percentile of absolute errors)
    abs_err = np.abs(y_true - y_pred)
    u95 = np.percentile(abs_err, 95)

    # MARD (Mean Absolute Relative Deviation)
    rel_err = np.abs((y_true - y_pred) / (y_true + 1e-8))
    mard = rel_err.mean()

    return {"R2": r2, "RMSE": rmse, "RAE": rae, "U95": u95, "MARD": mard}

metrics = calc_metrics(y, y_pred)

print("\n=== FINAL PERFORMANCE (on whole data) ===")
print(f"Run time          : {runtime:.2f} s")
print(f"Best RAE (CV)     : {best_rae:.6f}")
for k, v in metrics.items():
    print(f"{k:5s} : {v:.6f}")

# --------------------------------------------------------------
#  7. Save hyper-parameter history (optional)
# --------------------------------------------------------------
hist_df = pd.DataFrame(hp_history,
                       columns=[f"param_{i}" for i in range(dim)])
hist_df.to_excel("DOA_hyperparameters_history.xlsx", index_label="iteration")
print("\nHyper-parameter history saved to 'DOA_hyperparameters_history.xlsx'")