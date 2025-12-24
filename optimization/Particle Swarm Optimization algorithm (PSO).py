import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Load sample dataset (California Housing - regression task)
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

# Scale features (important for SVR)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Particle Swarm Optimization (PSO) implementation
def pso_optimize(objective_func, lb, ub, n_particles=30, max_iter=100, w=0.7, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization (PSO) for minimizing an objective function.
    Classic PSO with inertia weight and cognitive/social components.
    """
    D = len(lb)
    # Initialize particles and velocities
    particles = lb + np.random.rand(n_particles, D) * (ub - lb)
    velocities = np.random.uniform(-1, 1, (n_particles, D)) * (ub - lb)

    # Personal best and global best
    pbest = particles.copy()
    pbest_fitness = np.array([objective_func(p) for p in pbest])
    gbest_idx = np.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_fitness[gbest_idx]

    print(f"Iteration 0: Best score = {gbest_score:.4f}")

    for iter in range(1, max_iter + 1):
        for i in range(n_particles):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)

            # Velocity update
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))

            # Position update
            particles[i] += velocities[i]

            # Boundary handling
            particles[i] = np.clip(particles[i], lb, ub)

            # Evaluate
            score = objective_func(particles[i])

            # Update personal best
            if score < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = score

                # Update global best
                if score < gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = score
                    print(f"Iteration {iter}: Best params = {gbest}, Best score = {gbest_score:.4f}")

        # Linearly decrease inertia weight
        w = 0.9 - (0.9 - 0.4) * (iter / max_iter)

    return gbest, gbest_score


# Objective function for Support Vector Regression (SVR)
def objective_svr(params):
    """
    Objective for SVR.
    Params: [C, gamma, kernel_index (0: rbf, 1: poly, 2: linear)]
    """
    C = params[0]
    gamma = params[1]
    kernel_idx = int(params[2])
    kernel = ['rbf', 'poly', 'linear'][kernel_idx]

    model = SVR(
        C=C, gamma=gamma, kernel=kernel
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score  # Minimize MSE (negative because we minimize error)


# Objective function for Decision Trees (DT)
def objective_dt(params):
    """
    Objective for DecisionTreeRegressor.
    Params: [max_depth, min_samples_split, min_samples_leaf]
    """
    md = int(params[0])
    mss = int(params[1])
    msl = int(params[2])

    model = DecisionTreeRegressor(
        max_depth=md, min_samples_split=mss, min_samples_leaf=msl, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Hyperparameter bounds for SVR
lb_svr = np.array([1.0, 0.001, 0.0])
ub_svr = np.array([1000.0, 1.0, 2.0])

# Hyperparameter bounds for DT
lb_dt = np.array([3.0, 2.0, 1.0])
ub_dt = np.array([20.0, 20.0, 10.0])


# Optimize SVR using PSO
print("Optimizing SVR with PSO...")
best_params_svr, best_score_svr = pso_optimize(
    objective_svr, lb_svr, ub_svr, n_particles=25, max_iter=60
)
print(
    f"Best SVR params: C={best_params_svr[0]:.4f}, gamma={best_params_svr[1]:.4f}, "
    f"kernel={['rbf', 'poly', 'linear'][int(best_params_svr[2])]}"
)
print(f"Best CV MSE: {best_score_svr:.4f}")

# Train final SVR model and evaluate on test set
kernel = ['rbf', 'poly', 'linear'][int(best_params_svr[2])]
svr_final = SVR(
    C=best_params_svr[0],
    gamma=best_params_svr[1],
    kernel=kernel
)
svr_final.fit(X_train, y_train)
y_pred_svr = svr_final.predict(X_test)
test_mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"Test MSE for SVR: {test_mse_svr:.4f}\n")


# Optimize Decision Trees using PSO
print("Optimizing Decision Trees (DT) with PSO...")
best_params_dt, best_score_dt = pso_optimize(
    objective_dt, lb_dt, ub_dt, n_particles=25, max_iter=60
)
print(
    f"Best DT params: max_depth={int(best_params_dt[0])}, "
    f"min_samples_split={int(best_params_dt[1])}, min_samples_leaf={int(best_params_dt[2])}"
)
print(f"Best CV MSE: {best_score_dt:.4f}")

# Train final DT model and evaluate on test set
dt_final = DecisionTreeRegressor(
    max_depth=int(best_params_dt[0]),
    min_samples_split=int(best_params_dt[1]),
    min_samples_leaf=int(best_params_dt[2]),
    random_state=42
)
dt_final.fit(X_train, y_train)
y_pred_dt = dt_final.predict(X_test)
test_mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f"Test MSE for DT: {test_mse_dt:.4f}")