import numpy as np


def hoa_optimizer(
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
):
    # Ensure lb/ub are arrays
    lb = np.array(lb) if np.iterable(lb) else np.full(dim, lb)
    ub = np.array(ub) if np.iterable(ub) else np.full(dim, ub)

    # Initialize population and fitness
    X = lb + (ub - lb) * np.random.rand(n_agents, dim)
    fitness_f = np.zeros(n_agents)
    best_fitness = np.inf
    best_position = np.zeros(dim)
    convergence = [best_fitness]

    # Evaluate initial population
    for i in range(n_agents):
        fitness_f[i] = objective_function(X[i], X_train, y_train, X_test, y_test)
        if fitness_f[i] < best_fitness:
            best_fitness = fitness_f[i]
            best_position = X[i].copy()

    # Main loop
    for t in range(1, max_iter + 1):
        # Get global best so far
        ind = np.argmin(fitness_f)
        Xbest = X[ind, :].copy()

        for i in range(n_agents):
            Xini = X[i, :].copy()

            # Hiking parameters
            theta = np.random.randint(0, 51)  # elevation angle
            s = np.tan(np.radians(theta))  # slope
            SF = np.random.randint(1, 3)  # sweep factor (1 or 2)
            Vel = 6 * np.exp(
                -3.5 * np.abs(s + 0.05)
            )  # Tobler's Hiking Function velocity

            # Update position
            newVel = Vel + np.random.rand(dim) * (Xbest - SF * Xini)
            Xnew = Xini + newVel
            Xnew = np.minimum(ub, np.maximum(lb, Xnew))  # enforce bounds

            # Evaluate
            fnew = objective_function(Xnew, X_train, y_train, X_test, y_test)

            # Greedy selection
            if fnew < fitness_f[i]:
                X[i, :] = Xnew
                fitness_f[i] = fnew
                if fnew < best_fitness:
                    best_fitness = fnew
                    best_position = Xnew.copy()

        convergence.append(best_fitness)
        print(f"🥾 Iter {t}/{max_iter} - Best RMSE: {best_fitness:.5f}")

    return best_position, best_fitness, convergence
