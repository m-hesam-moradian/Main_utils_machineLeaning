import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# Load sample classification dataset for demonstration
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Bayesian Optimization Algorithm (BOA) implementation
def boa_optimize(objective_func, lb, ub, n_init=10, n_iter=50):
    """
    Bayesian Optimization Algorithm for minimizing an objective function.
    
    Uses Gaussian Process as surrogate and Expected Improvement as acquisition function.
    
    Parameters:
    - objective_func: function that takes a 1D array of parameters and returns a scalar value to minimize.
    - lb: lower bounds (array of length D)
    - ub: upper bounds (array of length D)
    - n_init: number of initial random points
    - n_iter: number of optimization iterations
    
    Returns:
    - best_params: optimized parameters
    - best_score: best objective value
    """
    D = len(lb)
    
    # Initialize random points
    X_sample = np.random.uniform(lb, ub, (n_init, D))
    y_sample = np.array([objective_func(x) for x in X_sample])
    
    # Define GP with Matern kernel
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5, random_state=42)
    
    # Expected Improvement acquisition function
    def expected_improvement(X, X_sample, y_sample, gp, xi=0.01):
        mu, sigma = gp.predict(X, return_std=True)
        mu_sample_opt = np.min(y_sample)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return -ei  # For minimization
    
    from scipy.optimize import minimize
    from scipy.stats import norm
    
    for i in range(n_iter):
        # Fit GP to samples
        gp.fit(X_sample, y_sample)
        
        # Find next point by minimizing the acquisition function
        def acq(x):
            x = x.reshape(1, -1)
            return expected_improvement(x, X_sample, y_sample, gp)
        
        res = minimize(acq, x0=np.random.uniform(lb, ub, D), bounds=list(zip(lb, ub)), method='L-BFGS-B')
        next_x = res.x
        
        # Evaluate objective
        next_y = objective_func(next_x)
        
        # Add to samples
        X_sample = np.vstack((X_sample, next_x))
        y_sample = np.append(y_sample, next_y)
    
    best_idx = np.argmin(y_sample)
    best_params = X_sample[best_idx]
    best_score = y_sample[best_idx]
    
    return best_params, best_score

# Objective function for Extra Trees Classifier (ETC)
def objective_etc(params):
    """Objective for ExtraTreesClassifier.
    Params: [n_estimators, max_depth, min_samples_split]
    """
    n_est = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])
    
    model = ExtraTreesClassifier(
        n_estimators=n_est,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return -score  # Minimize negative accuracy

# Objective function for Gradient Boosting Classifier (GBC)
def objective_gbc(params):
    """Objective for GradientBoostingClassifier.
    Params: [n_estimators, learning_rate, max_depth]
    """
    n_est = int(params[0])
    lr = params[1]
    max_depth = int(params[2])
    
    model = GradientBoostingClassifier(
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=max_depth,
        random_state=42
    )
    
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return -score

# Objective function for Random Forest Classifier (RFC)
def objective_rfc(params):
    """Objective for RandomForestClassifier.
    Params: [n_estimators, max_depth, min_samples_split]
    """
    n_est = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])
    
    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return -score

# Hyperparameter bounds
# ETC bounds
lb_etc = np.array([50, 3, 2])
ub_etc = np.array([200, 20, 10])

# GBC bounds
lb_gbc = np.array([50, 0.01, 3])
ub_gbc = np.array([200, 0.3, 10])

# RFC bounds
lb_rfc = np.array([50, 3, 2])
ub_rfc = np.array([200, 20, 10])

# Optimize ETC using BOA
print("Optimizing Extra Trees Classifier (ETC) with BOA...")
best_params_etc, best_score_etc = boa_optimize(
    objective_etc, lb_etc, ub_etc, n_init=10, n_iter=40
)
print(f"Best ETC params: n_estimators={int(best_params_etc[0])}, "
      f"max_depth={int(best_params_etc[1]) if best_params_etc[1] > 0 else 'None'}, min_samples_split={int(best_params_etc[2])}")
print(f"Best CV Negative Accuracy: {best_score_etc:.4f} (Accuracy: {-best_score_etc:.4f})")

# Train final ETC model
etc_final = ExtraTreesClassifier(
    n_estimators=int(best_params_etc[0]),
    max_depth=int(best_params_etc[1]) if best_params_etc[1] > 0 else None,
    min_samples_split=int(best_params_etc[2]),
    random_state=42
)
etc_final.fit(X_train, y_train)
y_pred_etc = etc_final.predict(X_test)
test_acc_etc = accuracy_score(y_test, y_pred_etc)
print(f"Test Accuracy for ETC: {test_acc_etc:.4f}\n")

# Optimize GBC using BOA
print("Optimizing Gradient Boosting Classifier (GBC) with BOA...")
best_params_gbc, best_score_gbc = boa_optimize(
    objective_gbc, lb_gbc, ub_gbc, n_init=10, n_iter=40
)
print(f"Best GBC params: n_estimators={int(best_params_gbc[0])}, "
      f"learning_rate={best_params_gbc[1]:.4f}, max_depth={int(best_params_gbc[2])}")
print(f"Best CV Negative Accuracy: {best_score_gbc:.4f} (Accuracy: {-best_score_gbc:.4f})")

# Train final GBC model
gbc_final = GradientBoostingClassifier(
    n_estimators=int(best_params_gbc[0]),
    learning_rate=best_params_gbc[1],
    max_depth=int(best_params_gbc[2]),
    random_state=42
)
gbc_final.fit(X_train, y_train)
y_pred_gbc = gbc_final.predict(X_test)
test_acc_gbc = accuracy_score(y_test, y_pred_gbc)
print(f"Test Accuracy for GBC: {test_acc_gbc:.4f}\n")

# Optimize RFC using BOA
print("Optimizing Random Forest Classifier (RFC) with BOA...")
best_params_rfc, best_score_rfc = boa_optimize(
    objective_rfc, lb_rfc, ub_rfc, n_init=10, n_iter=40
)
print(f"Best RFC params: n_estimators={int(best_params_rfc[0])}, "
      f"max_depth={int(best_params_rfc[1]) if best_params_rfc[1] > 0 else 'None'}, min_samples_split={int(best_params_rfc[2])}")
print(f"Best CV Negative Accuracy: {best_score_rfc:.4f} (Accuracy: {-best_score_rfc:.4f})")

# Train final RFC model
rfc_final = RandomForestClassifier(
    n_estimators=int(best_params_rfc[0]),
    max_depth=int(best_params_rfc[1]) if best_params_rfc[1] > 0 else None,
    min_samples_split=int(best_params_rfc[2]),
    random_state=42
)
rfc_final.fit(X_train, y_train)
y_pred_rfc = rfc_final.predict(X_test)
test_acc_rfc = accuracy_score(y_test, y_pred_rfc)
print(f"Test Accuracy for RFC: {test_acc_rfc:.4f}")

# Summary comparison
print("\n" + "="*60)
print("BOA OPTIMIZATION SUMMARY")
print("="*60)
models_summary = {
    "ETC": test_acc_etc,
    "GBC": test_acc_gbc,
    "RFC": test_acc_rfc
}
best_model = max(models_summary, key=models_summary.get)
print(f"Best performing model: {best_model} (Test Accuracy: {models_summary[best_model]:.4f})")
for model, acc in models_summary.items():
    print(f"{model:4s}: Test Accuracy = {acc:.4f}")