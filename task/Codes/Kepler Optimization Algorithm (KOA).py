# --------------------------------------------------------------
#  KOA + QR / ETR  (Excel loading as you requested)
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
#  KOA implementation (class)
# --------------------------------------------------------------
class KOA:
    def __init__(self, N=30, max_iter=200, lb=None, ub=None, dim=None, use_cv=True):
        self.N = N
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.use_cv = use_cv
        self.mu0 = 1.0
        self.gamma = 20.0
        self.epsilon = 1e-10
        self.pi = np.pi

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
        e = np.random.rand(self.N)
        T_period = np.abs(np.random.randn(self.N))
        fitness = np.array([self._fitness(ind) for ind in pop])

        # Initial best
        best_idx = np.argmin(fitness)
        gbest = pop[best_idx].copy()
        gbest_fit = fitness[best_idx]

        history = [gbest.copy()]  # iteration 0 (initial)

        start = time.time()
        for t in range(1, self.max_iter + 1):
            fit_s = gbest_fit
            worst = np.max(fitness)
            sum_diff = np.sum(fitness - worst)
            if sum_diff == 0:
                sum_diff = self.epsilon

            r2 = np.random.rand()
            M_s = r2 * (fit_s - worst) / sum_diff
            m = r2 * (fitness - worst) / sum_diff

            # Distance to sun (best)
            R = np.linalg.norm(pop - gbest, axis=1) ** 2
            min_R = np.min(R)
            max_R = np.max(R)
            if max_R > min_R:
                R_norm = (R - min_R) / (max_R - min_R)
            else:
                R_norm = np.zeros(self.N)

            mu = self.mu0 * np.exp(-self.gamma * t / self.max_iter)

            r1 = np.random.rand(self.N)
            F_g = e * mu * M_s * m / (R_norm**2 + self.epsilon + r1)

            # Semi-major axis
            r3_scalar = np.random.rand(self.N)
            a = r3_scalar * T_period**2 * mu * (M_s + m) / (4 * self.pi**2) * (1/3)

            # L
            L = np.sqrt(mu * (M_s + m) / (2 * R + self.epsilon) - 1 / (a + self.epsilon))

            # Random vectors (per dimension)
            r3 = np.random.rand(self.N, self.dim)
            r4 = np.random.rand(self.N, self.dim)
            r5 = np.random.rand(self.N, self.dim)
            r6 = np.random.rand(self.N, self.dim)

            U = (r5 > r6).astype(float)
            F_flag = (r4 <= 0.5).astype(float) * 2 - 1  # 1 or -1

            M = r3 * (1 - r4) + r4
            ddl = (1 - U) * M * L[:, np.newaxis]

            M_vec = r3 * (1 - r5) + r5
            U1 = (r5 > r4).astype(float)
            U2 = (r3 > r4).astype(float)

            # Random a and b
            a_idx = np.random.randint(0, self.N, self.N)
            b_idx = np.random.randint(0, self.N, self.N)
            X_a = pop[a_idx]
            X_b = pop[b_idx]

            # Velocity
            l = U * M_vec * L[:, np.newaxis]

            V = np.zeros_like(pop)
            mask = (R_norm <= 0.5)[:, np.newaxis]

            V[mask] = l[mask] * (2 * r4[mask] * (pop[mask] - X_b[mask])) + ddl[mask] * (X_a[mask] - X_b[mask]) + (1 - R_norm[mask, np.newaxis]) * F_flag[mask] * U1[mask] * r5[mask] * (self.ub - self.lb)

            V[~mask] = r4[~mask] * L[~mask, np.newaxis] * (X_a[~mask] - pop[~mask]) + (1 - R_norm[~mask, np.newaxis]) * F_flag[~mask] * U2[~mask] * r5[~mask] * r3[~mask] * (self.ub - self.lb)

            # Position update
            rn = np.random.randn(self.N, self.dim)
            X_new = pop + F_flag * V + (F_g[:, np.newaxis] + np.abs(rn)) * U * (gbest - pop)

            # Clip
            X_new = np.clip(X_new, self.lb, self.ub)

            # New fitness
            fitness_new = np.array([self._fitness(ind) for ind in X_new])

            # Elitism
            improve = fitness_new < fitness
            pop[improve] = X_new[improve]
            fitness[improve] = fitness_new[improve]

            # Update gbest
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

koa = KOA(N=30, max_iter=200, lb=lb, ub=ub, dim=dim, use_cv=True)
best_hp, best_rae, hp_history, runtime = koa.optimize()

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
hist_df.to_excel("KOA_hyperparameters_history.xlsx", index_label="iteration")
print("\nHyper-parameter history saved to 'KOA_hyperparameters_history.xlsx'")