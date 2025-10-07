# goa_hyperopt.py
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import math
import warnings

warnings.filterwarnings("ignore")


# ---------------------------
# Utility: Levy flight
# ---------------------------
def levy_flight(beta=1.5, dim=1):
    # Mantegna's algorithm
    sigma_u = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step


# ---------------------------
# GOA Optimizer (generic)
# ---------------------------
class GOAOptimizer:
    def __init__(
        self,
        func,  # objective: func(x_vector) -> score (higher=better)
        bounds,  # list of (min, max) for each dimension
        pop_size=20,
        max_iter=100,
        minimize=False,
        seed=None,
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.minimize = minimize
        if seed is not None:
            np.random.seed(seed)

    def _init_population(self):
        pop = np.random.rand(self.pop_size, self.dim)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        positions = lower + pop * (upper - lower)
        return positions

    def _clip(self, x):
        return np.minimum(np.maximum(x, self.bounds[:, 0]), self.bounds[:, 1])

    def optimize(self, verbose=False):
        # initialize
        positions = self._init_population()
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitness[i] = self.func(positions[i])
        # depending on minimize flag, convert to "higher is better"
        if self.minimize:
            fitness = -fitness

        best_idx = np.argmax(fitness)
        best_pos = positions[best_idx].copy()
        best_fit = fitness[best_idx]

        # main loop
        for t in range(1, self.max_iter + 1):
            # control parameter (like escape/pursuit intensity), decreasing with time
            a = 2 * (1 - (t / self.max_iter))  # linear decay from 2 -> 0

            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                # decision: exploration vs exploitation using random chance
                if np.random.rand() > 0.5:
                    # exploitation: move towards best (grazing peacefully)
                    step = a * r1 * (best_pos - positions[i])
                    new_pos = positions[i] + step
                else:
                    # exploration: outrun predator — large random moves (Levy flight)
                    lf = levy_flight(beta=1.5, dim=self.dim)
                    step = a * r2 * lf
                    new_pos = positions[i] + step * (positions[i] - best_pos)

                new_pos = self._clip(new_pos)
                new_fit = self.func(new_pos)
                if self.minimize:
                    new_fit = -new_fit

                # greedy selection
                if new_fit > fitness[i]:
                    positions[i] = new_pos
                    fitness[i] = new_fit
                    # update global best
                    if new_fit > best_fit:
                        best_fit = new_fit
                        best_pos = new_pos.copy()

            if verbose and (
                t % max(1, self.max_iter // 10) == 0 or t == 1 or t == self.max_iter
            ):
                reported = -best_fit if self.minimize else best_fit
                print(f"[GOA] iter {t}/{self.max_iter} best_score = {reported:.5f}")

        final_score = -best_fit if self.minimize else best_fit
        return best_pos, final_score


# ---------------------------
# Helpers: mapping vector -> hyperparameters
# ---------------------------
def map_vector_to_svc_params(vec):
    # vec: [v0, v1, v2] continuous in bounds defined below
    # map to:
    #   C in [1e-3, 1e3] (log scale)
    #   gamma in [1e-4, 1e1] (log scale)
    #   kernel index int in [0, len(kernels)-1]
    kernels = ["rbf", "linear", "poly", "sigmoid"]
    # expect scaled values: we'll receive direct continuous values for these bounds (see declarations)
    logC = vec[0]
    loggamma = vec[1]
    k_idx = int(np.round(vec[2]))
    k_idx = max(0, min(k_idx, len(kernels) - 1))
    degree = int(np.round(vec[3])) if vec.shape[0] > 3 else 3

    C = 10**logC  # logC is log10(C)
    gamma = 10**loggamma  # loggamma is log10(gamma)

    params = {
        "C": float(C),
        "gamma": float(gamma),
        "kernel": kernels[k_idx],
    }
    if params["kernel"] == "poly":
        params["degree"] = int(max(2, min(6, degree)))
    return params


def map_vector_to_lgb_params(vec):
    # map to LightGBM hyperparams:
    #   num_leaves [6, 256] -> int
    #   learning_rate [1e-3, 1.0] -> log space
    #   n_estimators [50, 1500] -> int
    #   max_depth [-1, 3, 15] -> map discrete via index
    max_depth_choices = [-1, 3, 5, 7, 10, 15]
    num_leaves = int(np.round(vec[0]))
    num_leaves = max(6, min(512, num_leaves))
    log_lr = vec[1]
    n_estimators = int(np.round(vec[2]))
    n_estimators = max(10, min(5000, n_estimators))
    md_idx = int(np.round(vec[3]))
    md_idx = max(0, min(len(max_depth_choices) - 1, md_idx))
    learning_rate = 10**log_lr

    params = {
        "num_leaves": int(num_leaves),
        "learning_rate": float(learning_rate),
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth_choices[md_idx]),
    }
    return params


# ---------------------------
# Objective wrappers
# ---------------------------
def make_svc_objective(X, y, cv=3, random_state=0, scoring="accuracy", n_jobs=1):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def obj(vec):
        params = map_vector_to_svc_params(vec)
        # build pipeline with scaling for SVC
        clf = make_pipeline(
            StandardScaler(),
            SVC(
                C=params["C"],
                kernel=params["kernel"],
                gamma=params.get("gamma", "scale"),
                degree=params.get("degree", 3),
                probability=False,
                random_state=random_state,
            ),
        )
        # use cross_val_score
        scores = cross_val_score(clf, X, y, cv=skf, scoring=scoring, n_jobs=n_jobs)
        return float(scores.mean())

    return obj


def make_lgb_objective(X, y, cv=3, random_state=0, scoring="accuracy", n_jobs=1):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def obj(vec):
        params = map_vector_to_lgb_params(vec)
        clf = lgb.LGBMClassifier(
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=random_state,
            n_jobs=n_jobs,
        )
        scores = cross_val_score(clf, X, y, cv=skf, scoring=scoring, n_jobs=n_jobs)
        return float(scores.mean())

    return obj


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Load demo dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # ----------------------
    # SVC optimization
    # ----------------------
    # Define bounds for vector:
    # log10(C) in [-3, 3]  -> C in [1e-3, 1e3]
    # log10(gamma) in [-4, 1] -> gamma in [1e-4, 1e1]
    # kernel index in [0, 3]  -> 4 choices
    # degree in [2, 6] (only used if kernel=poly)
    svc_bounds = [
        (-3.0, 3.0),  # log10(C)
        (-4.0, 1.0),  # log10(gamma)
        (0.0, 3.0),  # kernel index
        (2.0, 6.0),  # degree (integer)
    ]

    svc_obj = make_svc_objective(
        X, y, cv=4, random_state=42, scoring="accuracy", n_jobs=1
    )
    goa_svc = GOAOptimizer(
        func=svc_obj,
        bounds=svc_bounds,
        pop_size=20,
        max_iter=60,
        minimize=False,
        seed=42,
    )
    best_vec_svc, best_score_svc = goa_svc.optimize(verbose=True)
    best_params_svc = map_vector_to_svc_params(best_vec_svc)
    print("\nBest SVC score (CV mean accuracy):", best_score_svc)
    print("Best SVC params:", best_params_svc)

    # ----------------------
    # LightGBM optimization
    # ----------------------
    # num_leaves [6,512], log10(lr) in [-3, 0], n_estimators [50,1500], max_depth index [0..len-1]
    lgb_bounds = [
        (6, 512),  # num_leaves
        (-3.0, 0.0),  # log10(learning_rate)
        (50, 1500),  # n_estimators
        (0.0, 5.0),  # index for max_depth choices [-1,3,5,7,10,15]
    ]
    lgb_obj = make_lgb_objective(
        X, y, cv=4, random_state=42, scoring="accuracy", n_jobs=1
    )
    goa_lgb = GOAOptimizer(
        func=lgb_obj,
        bounds=lgb_bounds,
        pop_size=20,
        max_iter=80,
        minimize=False,
        seed=7,
    )
    best_vec_lgb, best_score_lgb = goa_lgb.optimize(verbose=True)
    best_params_lgb = map_vector_to_lgb_params(best_vec_lgb)
    print("\nBest LGB score (CV mean accuracy):", best_score_lgb)
    print("Best LGB params:", best_params_lgb)
