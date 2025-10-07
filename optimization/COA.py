"""
coa_hpo.py

Coati Optimization Algorithm (COA) for hyperparameter optimization (HPO).
Implements COA-inspired continuous optimizer and demonstrates tuning:
 - sklearn.svm.SVC
 - lightgbm.LGBMClassifier

References:
- COA File Exchange page (MATLAB) and original paper. (Used to implement COA-style operators.)
  https://www.mathworks.com/matlabcentral/fileexchange/116965-coa-coati-optimization-algorithm
  M. Dehghani et al.
"""

import numpy as np
import random
from copy import deepcopy
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")


# ---------- Utility: clamp ----------
def clamp(x, low, high):
    return np.minimum(np.maximum(x, low), high)


# ---------- COA Optimizer (continuous) ----------
class CoatiOptimizer:
    """
    Minimal COA-inspired continuous optimizer for hyperparameter search.
    This implementation is inspired by the COA metaheuristic (Dehghani et al.),
    adapted into a general continuous optimizer for bounded search spaces.

    Key parameters:
      - pop_size: number of coatis (search agents)
      - dim: dimensionality of the problem
      - lb, ub: bounds (arrays) for each dimension
      - max_iter: number of iterations
      - fitness_func: function(x) -> fitness (higher is better)
    """

    def __init__(
        self,
        fitness_func,
        dim,
        lb,
        ub,
        pop_size=30,
        max_iter=50,
        seed=None,
        elitism=True,
        verbose=False,
    ):
        self.fitness_func = fitness_func
        self.dim = dim
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.elitism = elitism
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def initialize(self):
        # initialize population uniformly in bounds
        X = np.random.rand(self.pop_size, self.dim) * (self.ub - self.lb) + self.lb
        return X

    def evaluate_population(self, X):
        # evaluate fitness in parallel
        fitness = np.array([self.fitness_func(x) for x in X])
        return fitness

    def run(self):
        # Initialization
        X = self.initialize()
        fitness = self.evaluate_population(X)
        best_idx = np.argmax(fitness)
        best_x = X[best_idx].copy()
        best_f = fitness[best_idx]
        history = [best_f]

        # main loop
        for t in range(self.max_iter):
            T = 1.0 - t / (self.max_iter - 1)  # cooling factor 1 -> 0
            new_X = X.copy()
            for i in range(self.pop_size):
                xi = X[i].copy()
                fi = fitness[i]
                # --- Hunting (exploration/exploitation mix) ---
                # choose mate (random better or random)
                mate_idx = np.random.randint(self.pop_size)
                mate = X[mate_idx]
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # exploration component: move toward/away from mate & best
                step1 = r1 * (mate - xi)
                step2 = r2 * (best_x - xi) * T

                # fleeing/predator escape: random jump scaled by T
                escape = (
                    (np.random.rand(self.dim) - 0.5) * (self.ub - self.lb) * (1 - T)
                )

                # combine
                candidate = xi + 0.5 * step1 + 0.8 * step2 + 0.4 * escape

                # small random local search around xi (local exploitation)
                local = (
                    (np.random.randn(self.dim) * 0.05) * (self.ub - self.lb) * (1 - T)
                )
                candidate += local

                # clamp to bounds
                candidate = clamp(candidate, self.lb, self.ub)

                # replace depending on fitness
                cand_f = self.fitness_func(candidate)
                if cand_f > fi:
                    new_X[i] = candidate
                else:
                    # allow occasional random exploration
                    if np.random.rand() < 0.15:
                        new_X[i] = self.lb + np.random.rand(self.dim) * (
                            self.ub - self.lb
                        )
                    else:
                        new_X[i] = xi

            # optional elitism: keep global best
            X = new_X
            fitness = self.evaluate_population(X)
            cur_best_idx = np.argmax(fitness)
            cur_best_f = fitness[cur_best_idx]
            cur_best_x = X[cur_best_idx].copy()
            if cur_best_f > best_f:
                best_f = cur_best_f
                best_x = cur_best_x.copy()

            # enforce elitism: maintain best in population
            if self.elitism:
                worst_idx = np.argmin(fitness)
                X[worst_idx] = best_x.copy()
                fitness[worst_idx] = best_f

            history.append(best_f)
            if self.verbose and (t % max(1, self.max_iter // 10) == 0):
                print(f"[COA] iter {t+1}/{self.max_iter} best_f={best_f:.4f}")

        return {"best_x": best_x, "best_f": best_f, "history": history}


# ---------- Hyperparameter helpers ----------
def decode_params_svc(x):
    """
    x is an array in continuous representation:
      x[0] -> log10(C) in [-3, 3]  => C = 10**x0
      x[1] -> log10(gamma) in [-4, 1] => gamma = 10**x1
      x[2] -> kernel continuous -> maps to {'rbf', 'poly', 'sigmoid'}
    """
    C = 10 ** float(x[0])
    gamma = 10 ** float(x[1])
    k_idx = int(np.clip(np.floor((x[2] - 0.0) / 1.0 * 3), 0, 2))
    kernels = ["rbf", "poly", "sigmoid"]
    kernel = kernels[k_idx]
    # degree only used if poly
    degree = int(np.clip(round(2 + (x[3] - 0.0) / 1.0 * 3), 2, 5)) if len(x) > 3 else 3
    return {"C": C, "gamma": gamma, "kernel": kernel, "degree": degree}


def decode_params_lgbm(x):
    """
    Example mapping (continuous to parameter):
      x[0] -> num_leaves in [8, 256]
      x[1] -> learning_rate in [-3, 0] as log10 -> lr = 10**x1
      x[2] -> max_depth in [3, 16]
      x[3] -> n_estimators in [50, 1000]
    """
    num_leaves = int(np.clip(round(x[0]), 8, 512))
    learning_rate = float(10 ** x[1])
    max_depth = int(np.clip(round(x[2]), -1, 20))
    n_estimators = int(np.clip(round(x[3]), 10, 2000))
    min_child_samples = int(np.clip(round(x[4]), 1, 500)) if len(x) > 4 else 20
    return {
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "min_child_samples": min_child_samples,
    }


# ---------- Fitness wrappers for classifiers ----------
def make_cv_fitness_classifier(
    model_builder, X, y, cv=3, scoring="accuracy", seed=None
):
    """
    model_builder: function(params_dict) -> estimator (sklearn-like)
    Returns a function f(x_continuous) -> mean_cv_score
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    def fitness(x):
        model = model_builder(x)
        # use cross_val_score with accuracy by default (higher is better)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    return fitness


# ---------- Demonstration on Iris dataset ----------
def demo_svc_coa(seed=42):
    data = load_iris()
    X = data.data
    y = data.target

    # search space:
    # x0 in [-3, 3] -> log10(C)
    # x1 in [-4, 1] -> log10(gamma)
    # x2 in [0, 3) -> kernel index continuous (0..3)
    # x3 in [0, 1] -> degree mapped to 2..5
    lb = np.array([-3.0, -4.0, 0.0, 0.0])
    ub = np.array([3.0, 1.0, 2.9999, 1.0])
    dim = len(lb)

    def model_builder(x):
        params = decode_params_svc(x)
        # create sklearn SVC with probability=False for speed
        return SVC(
            C=params["C"],
            gamma=params["gamma"],
            kernel=params["kernel"],
            degree=params["degree"],
            random_state=seed,
        )

    fitness = make_cv_fitness_classifier(
        model_builder, X, y, cv=5, scoring="accuracy", seed=seed
    )

    coa = CoatiOptimizer(
        fitness_func=fitness,
        dim=dim,
        lb=lb,
        ub=ub,
        pop_size=20,
        max_iter=40,
        seed=seed,
        verbose=True,
    )

    res = coa.run()
    best_params = decode_params_svc(res["best_x"])
    print("SVC best score:", res["best_f"])
    print("SVC best params:", best_params)
    return res, best_params


def demo_lgbm_coa(seed=42):
    data = load_iris()
    X = data.data
    y = data.target

    # search space mapping for LGBM
    # x0: num_leaves in [8, 512]
    # x1: log10(lr) in [-4, -1] => lr in [1e-4, 0.1]
    # x2: max_depth in [3, 12]
    # x3: n_estimators in [50, 500]
    # x4: min_child_samples in [1, 200]
    lb = np.array([8.0, -4.0, 3.0, 50.0, 1.0])
    ub = np.array([512.0, -0.3010, 12.0, 500.0, 200.0])
    dim = len(lb)

    def model_builder(x):
        params = decode_params_lgbm(x)
        # create LGBMClassifier (use n_jobs=1 to avoid nested parallel problems)
        return LGBMClassifier(
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            min_child_samples=params["min_child_samples"],
            random_state=seed,
            n_jobs=1,
        )

    fitness = make_cv_fitness_classifier(
        model_builder, X, y, cv=5, scoring="accuracy", seed=seed
    )

    coa = CoatiOptimizer(
        fitness_func=fitness,
        dim=dim,
        lb=lb,
        ub=ub,
        pop_size=24,
        max_iter=50,
        seed=seed,
        verbose=True,
    )

    res = coa.run()
    best_params = decode_params_lgbm(res["best_x"])
    print("LGBM best score:", res["best_f"])
    print("LGBM best params:", best_params)
    return res, best_params


if __name__ == "__main__":
    print("Demo COA -> SVC")
    svc_res, svc_params = demo_svc_coa(seed=123)

    print("\nDemo COA -> LGBM")
    lgbm_res, lgbm_params = demo_lgbm_coa(seed=123)
