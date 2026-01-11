import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ARDRegression  # Sparse Bayesian Regression (SBR)
from sklearn.preprocessing import StandardScaler
from scipy.special import gamma

# Load dataset from local Excel file
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
print(f"Loading dataset from {file_path}...")
df = pd.read_excel(file_path)

# Assume last column = target (continuous), others = features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Dataset loaded → Features shape: {X.shape}, Target shape: {y.shape}")

# Scale features (highly recommended)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================================
# 2. Echidna Optimization Algorithm (EOA) - based on provided reference (Arctic Puffin PDF)
# =====================================================
def levy_flight(D):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(D) * sigma
    v = np.random.randn(D)
    step = u / np.abs(v) ** (1 / beta)
    return step


def eoa_optimize(objective_func, lb, ub, N=25, T=70, F=0.5, C=0.5):
    """
    Echidna Optimization Algorithm (EOA)
    Based on the provided reference (Arctic Puffin behaviors):
    - Aerial flight (exploration): Aerial search + swoop predation
    - Underwater foraging (exploitation): gathering, intensifying, predator avoidance
    """
    start_time = time.time()
    D = len(lb)
    # Initialize population
    population = lb + np.random.rand(N, D) * (ub - lb)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_score = fitness[best_idx]
    print(f"Iteration 0: Best params = {best_params.round(4)}, Best MSE = {best_score:.4f}")

    convergence = [best_score]  # For MBE convergence plot

    for t in range(1, T + 1):
        B = 2 * np.log(1 / np.random.rand()) * (1 - t / T)  # Behavioral conversion factor

        for i in range(N):
            if B > C:  # Aerial flight phase (Exploration)
                # Strategy 1: Aerial search
                r = np.random.randint(0, N)
                while r == i:
                    r = np.random.randint(0, N)
                alpha = 2 * (1 - t / T)
                L = levy_flight(D)
                R = np.round(0.5 * (0.05 + np.random.rand())) * alpha
                Y = population[i] + (population[i] - population[r]) * L + R

                # Strategy 2: Swoop predation
                S = np.tan((np.random.rand() - 0.5) * np.pi)
                Z = Y * S

                # Choose better
                fit_Y = objective_func(Y)
                fit_Z = objective_func(Z)
                new_pos = Y if fit_Y < fit_Z else Z
                new_fit = min(fit_Y, fit_Z)
            else:  # Underwater foraging phase (Exploitation)
                r1, r2, r3 = np.random.choice(range(N), 3, replace=False)
                # Gathering foraging
                if np.random.rand() >= 0.5:
                    W = population[r1] + F * levy_flight(D) * (population[r2] - population[r3])
                else:
                    W = population[r1] + F * (population[r2] - population[r3])

                # Intensifying search
                f = 0.1 * (np.random.rand() - 1) * (T - t) / T
                Y = W * (1 + f)

                # Predator avoidance
                r4, r5 = np.random.choice(range(N), 2, replace=False)
                beta = np.random.rand()
                if np.random.rand() >= 0.5:
                    Z = population[i] + F * levy_flight(D) * (population[r4] - population[r5])
                else:
                    Z = population[i] + beta * (population[r4] - population[r5])

                # Choose best among W, Y, Z
                fit_W = objective_func(W)
                fit_Y = objective_func(Y)
                fit_Z = objective_func(Z)
                min_fit = min(fit_W, fit_Y, fit_Z)
                new_pos = W if min_fit == fit_W else (Y if min_fit == fit_Y else Z)
                new_fit = min_fit

            # Clip to bounds
            new_pos = np.clip(new_pos, lb, ub)

            # Update if better
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score:
                    best_params = new_pos.copy()
                    best_score = new_fit
                    print(f"Iteration {t:2d} → Best params = {best_params.round(4)}, MSE = {best_score:.4f}")

        convergence.append(best_score)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nTotal runtime: {runtime:.2f} seconds\n")

    # Convergence plot (MBE = Mean Best Error over iterations)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(convergence)), convergence, label='Best MSE per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Best MSE (MBE)')
    plt.title('Convergence Curve - Echidna Optimization Algorithm')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_params, best_score


# =====================================================
# Objective functions
# =====================================================

# Quantile Regression (QR)
def objective_qr(params):
    q = params[0]
    alpha = params[1]

    model = QuantileRegressor(quantile=q, alpha=alpha, solver='highs')

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Stochastic Gradient Boosting (SGB)
def objective_sgb(params):
    ne = int(params[0])
    lr = params[1]
    md = int(params[2])
    ss = params[3]

    model = GradientBoostingRegressor(
        n_estimators=ne, learning_rate=lr, max_depth=md, subsample=ss, random_state=42
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Sparse Bayesian Regression (SBR)
def objective_sbr(params):
    max_iter = int(params[0])
    tol = params[1]
    alpha_1 = params[2]
    alpha_2 = params[3]

    model = ARDRegression(
        max_iter=max_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2
    )

    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()
    return -score


# Hyperparameter bounds
lb_qr = np.array([0.1, 0.0])
ub_qr = np.array([0.9, 1.0])

lb_sgb = np.array([50, 0.01, 3, 0.5])
ub_sgb = np.array([300, 0.3, 15, 1.0])

lb_sbr = np.array([100, 1e-6, 1e-6, 1e-6])
ub_sbr = np.array([1000, 1e-2, 1e-2, 1e-2])


# =====================================================
# Run EOA on all models
# =====================================================

print("\n" + "="*90)
print("ECHIDNA OPTIMIZATION ALGORITHM (EOA) - 2024")
print("="*90)

# 1. Quantile Regression (QR)
print("\nOptimizing Quantile Regression (QR)...")
best_qr, score_qr = eoa_optimize(objective_qr, lb_qr, ub_qr, N=25, T=60)

print("\nBest QR parameters found:")
print(f"  quantile = {best_qr[0]:.4f}")
print(f"  alpha = {best_qr[1]:.4f}")
print(f"  Best CV MSE = {score_qr:.4f}")

qr_final = QuantileRegressor(quantile=best_qr[0], alpha=best_qr[1], solver='highs')
qr_final.fit(X_train, y_train)
test_mse_qr = mean_squared_error(y_test, qr_final.predict(X_test))
print(f"Final Test MSE (QR) = {test_mse_qr:.4f}\n")


# 2. Stochastic Gradient Boosting (SGB)
print("\nOptimizing Stochastic Gradient Boosting (SGB)...")
best_sgb, score_sgb = eoa_optimize(objective_sgb, lb_sgb, ub_sgb, N=25, T=60)

print("\nBest SGB parameters found:")
print(f"  n_estimators = {int(best_sgb[0])}")
print(f"  learning_rate = {best_sgb[1]:.4f}")
print(f"  max_depth = {int(best_sgb[2])}")
print(f"  subsample = {best_sgb[3]:.4f}")
print(f"  Best CV MSE = {score_sgb:.4f}")

sgb_final = GradientBoostingRegressor(
    n_estimators=int(best_sgb[0]),
    learning_rate=best_sgb[1],
    max_depth=int(best_sgb[2]),
    subsample=best_sgb[3],
    random_state=42
)
sgb_final.fit(X_train, y_train)
test_mse_sgb = mean_squared_error(y_test, sgb_final.predict(X_test))
print(f"Final Test MSE (SGB) = {test_mse_sgb:.4f}\n")


# 3. Sparse Bayesian Regression (SBR)
print("\nOptimizing Sparse Bayesian Regression (SBR)...")
best_sbr, score_sbr = eoa_optimize(objective_sbr, lb_sbr, ub_sbr, N=25, T=60)

print("\nBest SBR parameters found:")
print(f"  max_iter = {int(best_sbr[0])}")
print(f"  tol = {best_sbr[1]:.6f}")
print(f"  alpha_1 = {best_sbr[2]:.6f}")
print(f"  alpha_2 = {best_sbr[3]:.6f}")
print(f"  Best CV MSE = {score_sbr:.4f}")

sbr_final = ARDRegression(
    max_iter=int(best_sbr[0]),
    tol=best_sbr[1],
    alpha_1=best_sbr[2],
    alpha_2=best_sbr[3]
)
sbr_final.fit(X_train, y_train)
test_mse_sbr = mean_squared_error(y_test, sbr_final.predict(X_test))
print(f"Final Test MSE (SBR) = {test_mse_sbr:.4f}\n")