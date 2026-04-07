# --------------------------------------------------------------
#  AOA + QR / ETR  (Excel loading as you requested)
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
#  AOA implementation (class) - Addax Optimization Algorithm
# --------------------------------------------------------------
class AOA:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None, use_cv=True):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.use_cv = use_cv

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

        best_idx = np.argmin(fitness)
        gbest = pop[best_idx].copy()
        gbest_fit = fitness[best_idx]

        history = [gbest.copy()]  # iteration 0 (initial)

        start = time.time()
        for t in range(1, self.max_iter + 1):
            for i in range(self.N):
                # Phase 1: Foraging (exploration)
                # Candidate foraging areas: better solutions (F^k < F^i, k != i)
                candidates = [k for k in range(self.N) if fitness[k] < fitness[i] and k != i]
                if candidates:
                    sa_idx = np.random.choice(candidates)
                    S_A = pop[sa_idx]
                    r = np.random.rand(self.dim)
                    I = np.random.randint(1, 3, self.dim)  # 1 or 2
                    x_p1 = pop[i] + r * (S_A - I * pop[i])
                    x_p1 = np.clip(x_p1, self.lb, self.ub)
                    f_p1 = self._fitness(x_p1)
                    if f_p1 <= fitness[i]:
                        pop[i] = x_p1
                        fitness[i] = f_p1

                # Phase 2: Digging (exploitation) - applied after foraging
                r = np.random.rand(self.dim)
                delta = (1 - 2 * r) * (self.ub - self.lb) / t
                x_p2 = pop[i] + delta
                x_p2 = np.clip(x_p2, self.lb, self.ub)
                f_p2 = self._fitness(x_p2)
                if f_p2 <= fitness[i]:
                    pop[i] = x_p2
                    fitness[i] = f_p2

            # Update global best
            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < gbest_fit:
                gbest = pop[new_best_idx].copy()
                gbest_fit = fitness[new_best_idx]

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

aoa = AOA(N=30, max_iter=200, lb=lb, ub=ub, dim=dim, use_cv=True)
best_hp, best_rae, hp_history, runtime = aoa.optimize()

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
hist_df.to_excel("AOA_hyperparameters_history.xlsx", index_label="iteration")
print("\nHyper-parameter history saved to 'AOA_hyperparameters_history.xlsx'")