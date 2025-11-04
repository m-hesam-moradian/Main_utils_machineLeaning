# --------------------------------------------------------------
#  COA + QR / ETR  (Excel loading as you requested)
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
#  COA implementation (class)
# --------------------------------------------------------------
class COA:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None, use_cv=True):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.use_cv = use_cv
        self.T_max = 40.0
        self.F_max = 3.0

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

    # ---------- main optimisation loop ----------
    def optimize(self):
        np.random.seed(42)
        pop = self.lb + np.random.rand(self.N, self.dim) * (self.ub - self.lb)
        fitness = np.array([self._fitness(ind) for ind in pop])

        pbest = pop.copy()
        pbest_fit = fitness.copy()
        gbest_idx = np.argmin(fitness)
        gbest = pop[gbest_idx].copy()
        gbest_fit = fitness[gbest_idx]

        history = [gbest.copy()]          # iteration 0 (initial)

        start = time.time()
        for it in range(1, self.max_iter + 1):
            T = (it / self.max_iter) * self.T_max

            for i in range(self.N):
                if T > 30:                                 # competition
                    C = (pbest[i] + gbest) / 2
                    if np.random.rand() > 0.5:
                        pop[i] = C
                    else:
                        pop[i] = pop[np.random.randint(self.N)]
                else:                                      # foraging
                    F_loc = gbest
                    fi, ff = fitness[i], gbest_fit
                    diff = np.abs(fitness - ff)
                    max_diff = max(diff.max(), 1e-6)
                    S = self.F_max * np.abs(fi - ff) / max_diff
                    r = np.random.rand()
                    if S > 2:
                        trig = np.sin(np.pi * r) if np.random.rand() < 0.5 else np.cos(np.pi * r)
                        pop[i] = F_loc + F_loc * trig
                    else:
                        pop[i] = F_loc + r * (F_loc - pop[i])

                pop[i] = np.clip(pop[i], self.lb, self.ub)

            fitness = np.array([self._fitness(ind) for ind in pop])

            # update personal best
            improve = fitness < pbest_fit
            pbest[improve] = pop[improve]
            pbest_fit[improve] = fitness[improve]

            # update global best
            new_g_idx = np.argmin(fitness)
            if fitness[new_g_idx] < gbest_fit:
                gbest = pop[new_g_idx].copy()
                gbest_fit = fitness[new_g_idx]

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

coa = COA(N=30, max_iter=200, lb=lb, ub=ub, dim=dim, use_cv=True)
best_hp, best_rae, hp_history, runtime = coa.optimize()

# --------------------------------------------------------------
#  4. Show hyper-parameters for **all** iterations (first 10 + last)
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
hist_df.to_excel("COA_hyperparameters_history.xlsx", index_label="iteration")
print("\nHyper-parameter history saved to 'COA_hyperparameters_history.xlsx'")