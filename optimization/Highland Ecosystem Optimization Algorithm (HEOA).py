import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# Load sample dataset for demonstration (UCI Adult for classification)
print("Loading UCI Adult dataset...")
data = fetch_openml(name='adult', version=2, as_frame=True)
X, y = data.data, data.target
y = LabelEncoder().fit_transform(y)  # '>50K' → 1

# Preprocess: one-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

# Scale features (important for KNNC)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Highland Ecosystem Optimization Algorithm (HEOA) implementation
def heoa_optimize(objective_func, lb, ub, N=30, T=100):
    """
    Highland Ecosystem Optimization Algorithm for minimizing an objective function.
    Inspired by highland ecosystem dynamics: predator pursuit (exploitation), prey evasion (exploration), seasonal migration (spiral movement), alpine adaptation (refinement), and resource pressure (diversification).
    """
    start_time = time.time()
    D = len(lb)
    population = lb + np.random.rand(N, D) * (ub - lb)

    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(1, T + 1):
        a = 2 * (1 - t / T)

        # Sort indices by fitness (ascending, minimize)
        sorted_idx = np.argsort(fitness)
        Ge_idx = sorted_idx[-N//2:]  # Worse for exploration
        Gx_idx = sorted_idx[:N//2]  # Better for exploitation

        # Exploration phase
        for i in Ge_idx:
            if np.random.rand() < 0.5:
                # Basic exploration
                r1 = np.random.rand(D)
                r2 = np.random.rand(D)
                A = 2 * a * r1
                C = 2 * r2
                new_pos = best_params - A * np.abs(C * best_params - population[i])
            else:
                # Enhanced exploration
                rand_idx = np.random.choice(range(N), 3, replace=False)
                X_r1, X_r2, X_r3 = population[rand_idx]
                z = np.exp(-t / T)
                r = np.random.rand(D)
                weights = np.random.dirichlet(np.ones(3), size=1)[0]
                weighted_sum = weights[0] * X_r1 + weights[1] * X_r2 + weights[2] * X_r3
                new_pos = population[i] + z * (r * weighted_sum - population[i])

            # Secondary update
            r_sec = 2 * np.random.rand(D) - 1  # [-1,1]
            A_sec = np.random.uniform(0, 2, D)
            C_sec = np.random.rand(D)
            new_pos = new_pos + a * r_sec * (best_params - new_pos) + C_sec * np.abs(A_sec * best_params - new_pos)

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t}: Best params = {best_params}, Best error = {best_score:.4f}")

        # Exploitation phase
        for i in Gx_idx:
            if np.random.rand() < 0.5:
                # Sentry-guided update
                top_idx = sorted_idx[:3]
                X_s1, X_s2, X_s3 = population[top_idx]
                D1 = np.abs(X_s1 - population[i])
                D2 = np.abs(X_s2 - population[i])
                D3 = np.abs(X_s3 - population[i])
                r = np.random.rand(D)
                a1 = 2 * a * r
                a2 = 2 * a * r
                a3 = 2 * a * r
                new_pos = ( (X_s1 - a1 * D1) + (X_s2 - a2 * D2) + (X_s3 - a3 * D3) ) / 3
            else:
                # Leader-following
                r1 = np.random.rand(D)
                r2 = np.random.rand(D)
                A = 2 * a * r1
                C = 2 * r2
                new_pos = best_params - A * np.abs(C * best_params - population[i])

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate
            new_fit = objective_func(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t}: Best params = {best_params}, Best error = {best_score:.4f}")

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")

    return best_params, best_score


# Objective function for K-Nearest Neighbors Classification (KNNC)
def objective_knnc(params):
    """
    Objective for KNeighborsClassifier.
    Params: [n_neighbors, weights_index (0: uniform, 1: distance), p]
    """
    nn = int(params[0])
    wi = int(params[1])
    p = int(params[2])

    weights = ['uniform', 'distance'][wi]

    model = KNeighborsClassifier(
        n_neighbors=nn, weights=weights, p=p
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score  # Minimize 1 - accuracy


# Objective function for Adaptive Gradient Boosting Classification (ADAC)
def objective_adac(params):
    """
    Objective for AdaBoostClassifier.
    Params: [n_estimators, learning_rate]
    """
    ne = int(params[0])
    lr = params[1]

    model = AdaBoostClassifier(
        n_estimators=ne, learning_rate=lr, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()
    return 1 - score


# Hyperparameter bounds for KNNC
lb_knnc = np.array([3, 0, 1])
ub_knnc = np.array([30, 1, 2])

# Hyperparameter bounds for ADAC
lb_adac = np.array([50, 0.01])
ub_adac = np.array([200, 1.0])

# Optimize KNNC using HEOA
print("Optimizing KNNC with HEOA...")
best_params_knnc, best_error_knnc = heoa_optimize(
    objective_knnc, lb_knnc, ub_knnc, N=20, T=50
)
print(
    f"Best KNNC params: n_neighbors={int(best_params_knnc[0])}, weights={['uniform', 'distance'][int(best_params_knnc[1])]}, p={int(best_params_knnc[2])}"
)
print(f"Best CV error: {best_error_knnc:.4f}")

# Train final KNNC model and evaluate on test set
weights = ['uniform', 'distance'][int(best_params_knnc[1])]
knnc_final = KNeighborsClassifier(
    n_neighbors=int(best_params_knnc[0]),
    weights=weights,
    p=int(best_params_knnc[2])
)
knnc_final.fit(X_train, y_train)
y_pred_knnc = knnc_final.predict(X_test)
test_acc_knnc = accuracy_score(y_test, y_pred_knnc)
print(f"Test accuracy for KNNC: {test_acc_knnc:.4f}\n")

# Optimize ADAC using HEOA
print("Optimizing ADAC with HEOA...")
best_params_adac, best_error_adac = heoa_optimize(
    objective_adac, lb_adac, ub_adac, N=20, T=50
)
print(
    f"Best ADAC params: n_estimators={int(best_params_adac[0])}, learning_rate={best_params_adac[1]:.4f}"
)
print(f"Best CV error: {best_error_adac:.4f}")

# Train final ADAC model and evaluate on test set
adac_final = AdaBoostClassifier(
    n_estimators=int(best_params_adac[0]),
    learning_rate=best_params_adac[1],
    random_state=42
)
adac_final.fit(X_train, y_train)
y_pred_adac = adac_final.predict(X_test)
test_acc_adac = accuracy_score(y_test, y_pred_adac)
print(f"Test accuracy for ADAC: {test_acc_adac:.4f}")