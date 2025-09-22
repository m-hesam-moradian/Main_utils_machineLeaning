import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def hgso_optimizer(
    objective_function,
    lb,
    ub,
    dim,
    n_agents,
    max_iter,
    X_train,
    y_train,
    X_test,
    y_test,
    n_types=3,  # Default number of gas types
):
    # Ensure lb/ub are arrays
    lb = np.array(lb) if np.iterable(lb) else np.full(dim, lb)
    ub = np.array(ub) if np.iterable(ub) else np.full(dim, ub)

    # HGSO constants (from MATLAB code)
    l1 = 5e-3
    l2 = 100
    l3 = 1e-2
    alpha = 1
    beta = 1
    M1 = 0.127  # Messy value for worst agents
    M2 = 0.243  # Messy value for worst agents

    # Initialize parameters
    K = l1 * np.random.rand(n_types, 1)  # Henry's constant
    P = l2 * np.random.rand(n_agents, 1)  # Partial pressure
    C = l3 * np.random.rand(n_types, 1)  # Solubility constant

    # Initialize population
    X = lb + (ub - lb) * np.random.rand(n_agents, dim)
    fitness = np.zeros(n_agents)
    best_fitness = np.inf
    best_position = np.zeros(dim)
    convergence = [best_fitness]

    # Divide agents into n_types clusters
    n_per_type = n_agents // n_types
    groups = [X[i * n_per_type : (i + 1) * n_per_type] for i in range(n_types)]
    if n_agents % n_types:
        groups[-1] = np.vstack(
            (groups[-1], X[n_types * n_per_type :])
        )  # Handle remainder

    # Initialize best positions and fitness per group
    best_fit = np.full(n_types, np.inf)
    best_pos = [np.zeros(dim) for _ in range(n_types)]

    # Evaluate initial population
    for i in range(n_types):
        for j in range(groups[i].shape[0]):
            fitness[j + i * n_per_type] = objective_function(
                groups[i][j], X_train, y_train, X_test, y_test
            )
            if fitness[j + i * n_per_type] < best_fit[i]:
                best_fit[i] = fitness[j + i * n_per_type]
                best_pos[i] = groups[i][j].copy()
        if best_fit[i] < best_fitness:
            best_fitness = best_fit[i]
            best_position = best_pos[i].copy()

    # Main loop
    for t in range(1, max_iter + 1):
        # Update variables (eq. 7 in HGSO paper)
        T = t / max_iter
        K = K * np.exp(-C * (1 / np.exp(-T) - 1))  # Update Henry's constant
        P = P * np.exp(-C * (1 / np.exp(-T) - 1))  # Update partial pressure
        S = K * P  # Solubility (eq. 7)

        # Update positions for each group
        for i in range(n_types):
            group = groups[i]
            new_group = np.zeros_like(group)
            for j in range(group.shape[0]):
                idx = i * n_per_type + j if i * n_per_type + j < n_agents else -1
                if idx == -1:
                    break
                # Update position (eq. 8-9)
                F = -1 if np.random.rand() < 0.5 else 1  # Random direction
                r = np.random.rand()
                new_group[j] = (
                    group[j]
                    + F * r * alpha * S[i] * (best_pos[i] - group[j])
                    + F * r * beta * (best_position - group[j])
                )
                new_group[j] = np.minimum(
                    ub, np.maximum(lb, new_group[j])
                )  # Enforce bounds

                # Evaluate new position
                f_new = objective_function(
                    new_group[j], X_train, y_train, X_test, y_test
                )
                if f_new < fitness[idx]:
                    group[j] = new_group[j]
                    fitness[idx] = f_new
                    if f_new < best_fit[i]:
                        best_fit[i] = f_new
                        best_pos[i] = new_group[j].copy()
                        if f_new < best_fitness:
                            best_fitness = f_new
                            best_position = new_group[j].copy()

            # Update worst agents (eq. 11)
            n_worst = int(M1 * group.shape[0])
            worst_indices = np.argsort(fitness[i * n_per_type : (i + 1) * n_per_type])[
                -n_worst:
            ]
            for j in worst_indices:
                group[j] = lb + (ub - lb) * np.random.rand(dim)
                idx = i * n_per_type + j if i * n_per_type + j < n_agents else -1
                if idx != -1:
                    fitness[idx] = objective_function(
                        group[j], X_train, y_train, X_test, y_test
                    )

            groups[i] = group

        convergence.append(best_fitness)
        print(f"🔬 Iter {t}/{max_iter} - Best RMSE: {best_fitness:.5f}")

    return best_position, best_fitness, convergence
