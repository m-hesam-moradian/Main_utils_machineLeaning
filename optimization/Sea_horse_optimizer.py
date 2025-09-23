import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
import matplotlib.pyplot as plt
from Metrics_regression import getAllMetric  # Assume your custom metrics

# Dataset loading
sheet_name = "Data After K-FOLD"
df = pd.read_excel(
    r"D:\ML\Main_utils\Task\GLEMETA_MADDPG_Final_IoT_MEC_UAV_Dataset.xlsx",
    sheet_name=sheet_name,
)
target_column = "offload_ratio"

# Features and Target
X = df.drop(columns=[target_column])
y = df[target_column]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold setup for CV
kf = KFold(n_splits=5, shuffle=False)
rmse_scorer = make_scorer(
    mean_squared_error, squared=False
)  # Use RMSE as fitness (minimize)

# Bounds for hyperparameters: alpha (log scale 0.001-10), l1_ratio (0-1)
lb = np.array([0.001, 0.0])  # Lower bounds
ub = np.array([10.0, 1.0])  # Upper bounds
dim = 2  # Number of hyperparameters


# Sea Horse Optimizer (SHO) Implementation
def seahorse_optimizer(obj_func, lb, ub, dim, pop_size=30, max_iter=200, verbose=True):
    """
    Sea Horse Optimizer for hyperparameter tuning.
    Inspired by sea horse behaviors: movement, predation, reproduction.
    """
    # Initialize population
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    # Convergence tracking
    convergence = np.zeros(max_iter)
    convergence[0] = best_fit

    for t in range(1, max_iter):
        for i in range(pop_size):
            # Movement Phase (Exploration/Exploitation)
            r = np.random.rand()
            theta = 2 * np.pi * np.random.rand()
            # Spiral floating (inferred equation)
            V1 = np.random.randn(dim) * 0.1  # Small vortex influence
            V2 = np.random.randn(dim) * 0.1
            temp_pos = pop[i] + r * (np.cos(theta) * V1 + np.sin(theta) * V2)

            # Wave drifting (alternative mode, random selection)
            if np.random.rand() < 0.5:
                wave_dir = np.random.randn(dim) * 0.05
                temp_pos = pop[i] + wave_dir

            # Bound check
            temp_pos = np.clip(temp_pos, lb, ub)

            # Predation Phase (Move towards best if successful)
            p_success = 0.8  # Probability of successful predation
            if np.random.rand() < p_success:
                alpha = 2 * np.random.rand() - 1  # Step size [-1,1]
                temp_pos = temp_pos + alpha * (best_pos - temp_pos)
                temp_pos = np.clip(temp_pos, lb, ub)

            # Reproduction Phase (Generate offspring, weighted towards current)
            if np.random.rand() < 0.3:  # Reproduction probability
                beta = np.random.rand()
                rand_pos = np.random.uniform(lb, ub, dim)
                temp_pos = beta * pop[i] + (1 - beta) * rand_pos
                temp_pos = np.clip(temp_pos, lb, ub)

            # Evaluate
            new_fit = obj_func(temp_pos)

            # Update if better
            if new_fit < fitness[i]:
                pop[i] = temp_pos
                fitness[i] = new_fit
                if new_fit < best_fit:
                    best_pos = temp_pos.copy()
                    best_fit = new_fit

        convergence[t] = best_fit
        if verbose and t % 50 == 0:
            print(f"Iteration {t}: Best Fitness = {best_fit:.4f}")

    return best_pos, best_fit, convergence, pop, fitness


# Objective function: Negative R2 (maximize R2 -> minimize -R2), but using RMSE for consistency
def objective(params):
    alpha, l1_ratio = params
    if l1_ratio < 0 or l1_ratio > 1:
        return np.inf
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
    return -np.mean(scores)  # Minimize negative mean R2


# Run SHO for optimization
best_params, best_fit, convergence, _, _ = seahorse_optimizer(
    objective, lb, ub, dim, pop_size=30, max_iter=200, verbose=True
)

# Best hyperparameters
best_alpha, best_l1_ratio = best_params
print(f"\nFinal Hyperparameters: alpha={best_alpha:.4f}, l1_ratio={best_l1_ratio:.4f}")
print(f"Best CV R2: {-best_fit:.4f}")

# Plot Convergence
plt.figure(figsize=(10, 6))
plt.plot(convergence, "b-", linewidth=2)
plt.title("SHO Convergence Curve (200 Iterations)")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (-Mean R2)")
plt.grid(True)
plt.show()

# Train final model with best params
final_model = ElasticNet(
    alpha=best_alpha,
    l1_ratio=best_l1_ratio,
    fit_intercept=True,
    max_iter=1000,
    tol=0.0001,
    random_state=42,
    selection="cyclic",
)
final_model.fit(X_scaled, y)  # Fit on full data for metrics

# Predictions for metrics (like your original)
y_pred_all = final_model.predict(X_scaled)

# Your custom metrics
R, RMSE, MAE, RSE, SMAPE = getAllMetric(y, y_pred_all)

# Train-Test Split for additional metrics (as in original)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

final_model.fit(X_train_scaled, y_train)
y_pred_train = final_model.predict(X_train_scaled)
y_pred_test = final_model.predict(X_test_scaled)

# Split test predictions
mid_index = len(y_pred_test) // 2
y_test_first_half = y_test.iloc[:mid_index]
y_test_second_half = y_test.iloc[mid_index:]
y_pred_test_first_half = y_pred_test[:mid_index]
y_pred_test_second_half = y_pred_test[mid_index:]

# Metrics table
metrics_data = {"Set": [], "R2": [], "RMSE": [], "MAE": [], "RSE": [], "SMAPE": []}
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test_first_half, y_pred_test_first_half),
    ("Test-Value", y_test_second_half, y_pred_test_second_half),
]

for name, y_true, y_pred in sets:
    R_set, RMSE_set, MAE_set, RSE_set, SMAPE_set = getAllMetric(y_true, y_pred)
    metrics_data["Set"].append(name)
    metrics_data["R2"].append(R_set)
    metrics_data["RMSE"].append(RMSE_set)
    metrics_data["MAE"].append(MAE_set)
    metrics_data["RSE"].append(RSE_set)
    metrics_data["SMAPE"].append(SMAPE_set)

metrics_df = pd.DataFrame(metrics_data)

print("\n📋 Performance Metrics Table (Optimized ElasticNet):")
print(metrics_df)
