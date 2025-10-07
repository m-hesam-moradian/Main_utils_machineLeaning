import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import lightgbm as lgb


def koa_optimize(obj_func, lb, ub, Np=30, MaxIt=100, dim=None):
    """
    Kepler Optimization Algorithm (KOA) for hyperparameter optimization.

    Parameters:
    - obj_func: objective function to minimize (takes param vector, returns scalar)
    - lb, ub: lower and upper bounds (arrays of length dim)
    - Np: population size
    - MaxIt: maximum iterations
    - dim: dimension (inferred from lb if None)

    Returns:
    - best_params: optimal parameters
    - best_score: optimal objective value
    """
    if dim is None:
        dim = len(lb)
    # Initialize population
    X = np.random.uniform(lb, ub, (Np, dim))
    fitness = np.full(Np, np.inf)
    for i in range(Np):
        fitness[i] = obj_func(X[i])
    # Initial best
    best_idx = np.argmin(fitness)
    X_s = X[best_idx].copy()
    best_fit = fitness[best_idx]
    eps = 1e-10
    for t in range(1, MaxIt + 1):
        mu_t = 0.1 * np.exp(-15 * t / MaxIt)
        a2 = np.abs(np.sin(2 * np.pi * t / MaxIt))
        worst = np.max(fitness)
        denom = worst - best_fit + eps
        for i in range(Np):
            # Distance to sun
            diff = X[i] - X_s
            R_i = np.linalg.norm(diff)
            if R_i < eps:
                R_i = eps
            # Normalized distances
            dists = np.linalg.norm(X - X_s, axis=1)
            max_R = np.max(dists) + eps
            R_norm = R_i / max_R
            # Orbital period T_i (absolute normal)
            T_i = np.abs(np.random.normal(1, 0.5))
            # Semi-major axis
            a_i = ((T_i**2 * mu_t) / (4 * np.pi**2)) ** (1 / 3) * R_norm
            if a_i < eps:
                a_i = eps
            # Orbital velocity component (vis-viva)
            v_orbital = mu_t * (2 / (R_i + eps) - 1 / a_i)
            # Direction flag
            F_dir = np.random.choice([-1, 1])
            # Random other solutions
            idxs = [j for j in range(Np) if j != i]
            if len(idxs) > 0:
                idx_a = np.random.choice(idxs)
                idx_b = np.random.choice(idxs)
                X_a = X[idx_a]
                X_b = X[idx_b]
            else:
                X_a = X_s
                X_b = X_s
            # Random coefficients
            r1, r2, r3, r4, r5 = np.random.rand(5)
            # Velocity components
            term1 = F_dir * v_orbital * diff
            term2 = r1 * (X_a - X_b)
            term3 = r2 * diff
            term4 = r3 * (-diff)
            term5 = r4 * (X[i] - X_a)
            term6 = r5 * (X_b - X[i])
            V = term1 + term2 + term3 + term4 + term5 + term6
            # Normalized masses (lower fitness = higher mass)
            mn_i = (worst - fitness[i]) / denom
            Mn_s = 1.0
            # Normalized distance for gravity
            Rn_i = R_i / max_R
            e_i = np.random.rand()
            # Gravitational force
            Fg_i = mu_t * Mn_s * mn_i * e_i / (Rn_i**2 + eps)
            # Gravitational term
            grav_term = F_dir * Fg_i * (-diff)
            # New position
            X_new = X[i] + V + grav_term
            X_new = np.clip(X_new, lb, ub)
            # Evaluate
            fit_new = obj_func(X_new)
            # Elitism
            if fit_new < fitness[i]:
                X[i] = X_new
                fitness[i] = fit_new
                if fit_new < best_fit:
                    best_fit = fit_new
                    X_s = X_new.copy()
            # Step 5: Cyclic distance adjustment (pull towards sun)
            pull = a2 * (-diff)
            X[i] += pull * 0.1  # Scaled to avoid excessive movement
            X[i] = np.clip(X[i], lb, ub)
            # Re-evaluate if changed
            if np.linalg.norm(X[i] - X_new) > eps:
                fit_new2 = obj_func(X[i])
                if fit_new2 < fitness[i]:
                    fitness[i] = fit_new2
                    if fit_new2 < best_fit:
                        best_fit = fit_new2
                        X_s = X[i].copy()
    return X_s, best_fit


# Example usage with Iris dataset
data = load_iris()
X_data, y_data = data.data, data.target


# For SVC (Support Vector Classifier)
def svc_objective(params):
    C = np.exp(params[0])
    gamma = np.exp(params[1])
    model = SVC(C=C, gamma=gamma, random_state=42)
    scores = cross_val_score(model, X_data, y_data, cv=5)
    return -np.mean(scores)  # Minimize negative accuracy


lb_svc = np.log([1e-3, 1e-3])
ub_svc = np.log([1e3, 1e3])
best_params_svc, best_score_svc = koa_optimize(
    svc_objective, lb_svc, ub_svc, Np=20, MaxIt=50
)
print("SVC Best params (C, gamma):", np.exp(best_params_svc))
print("SVC Best CV Score:", -best_score_svc)


# For LGBC (LightGBM Classifier)
def lgb_objective(params):
    learning_rate = params[0]
    num_leaves = int(np.clip(params[1], 10, 200))
    model = lgb.LGBMClassifier(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        n_estimators=100,
        verbose=-1,
        random_state=42,
    )
    scores = cross_val_score(model, X_data, y_data, cv=5)
    return -np.mean(scores)  # Minimize negative accuracy


lb_lgb = np.array([0.01, 10.0])
ub_lgb = np.array([0.3, 200.0])
best_params_lgb, best_score_lgb = koa_optimize(
    lgb_objective, lb_lgb, ub_lgb, Np=20, MaxIt=50
)
print("LGBC Best params (learning_rate, num_leaves):", best_params_lgb)
print("LGBC Best CV Score:", -best_score_lgb)
