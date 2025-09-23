import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
import matplotlib.pyplot as plt
from Metrics_regression import getAllMetric  # Assume your custom metrics

# Dataset loading
sheet_name = "Data after K-Fold (GBR & ANFIS)"
df = pd.read_excel(
    r"D:\ML\Main_utils\Task\Global_AI_Content_Impact_Dataset.xlsx",
    sheet_name=sheet_name,
)
target_column = "Market Share of AI Companies (%)"

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

# Bounds for hyperparameters: alpha (log scale 1e-4 to 1e4)
lb = np.array([1e-4])  # Lower bounds (dim=1 for alpha)
ub = np.array([1e4])  # Upper bounds
dim = 1  # Number of hyperparameters (alpha only)


# Zebra Optimization Algorithm (ZOA) Implementation (Adapted from AZOA)
def zebra_optimizer(obj_func, lb, ub, dim, pop_size=30, max_iter=200, verbose=True):
    """
    Zebra Optimization Algorithm for hyperparameter tuning.
    Adapted from American Zebra Optimization (AZOA): Groups, feeding (sin/cos updates),
    breeding (crossover), stallion updates, leadership transition.
    """
    # Parameters (from AZOA)
    P = 0.1  # Stallion probability
    pc = 0.1  # Crossover probability
    N_groups = int(pop_size * P)  # Number of groups (e.g., 3 for pop=30)
    group_size = pop_size // N_groups  # Zebras per group

    # Initialize population (zebras)
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    global_best = pop[best_idx].copy()  # WR: Water Reserves (global best)
    best_fit = fitness[best_idx]

    # Convergence tracking
    convergence = np.zeros(max_iter)
    convergence[0] = best_fit

    for t in range(1, max_iter + 1):
        # Adaptive parameters
        R2 = 1 - t * (1 / max_iter)
        R5 = R2  # Same as R2

        # Divide into groups (simplified: first N_groups * group_size)
        for g in range(N_groups):
            group_start = g * group_size
            group_end = (g + 1) * group_size
            group_pop = pop[group_start:group_end]
            group_fit = fitness[group_start:group_end]

            # Stallion (best in group)
            stallion_idx = np.argmin(group_fit)
            stallion_pos = group_pop[stallion_idx].copy()

            # Phase 2: Feeding Activity (Zebras update towards stallion)
            for i in range(group_size):
                if i == stallion_idx:
                    continue  # Skip stallion
                R1 = np.random.rand(dim)
                R3 = np.random.rand()
                if R3 < 0.5:
                    temp_pos = (
                        2 * R1 * np.sin(2 * np.pi * R2) * (stallion_pos - group_pop[i])
                        + stallion_pos
                    )
                else:
                    temp_pos = (
                        2 * R1 * np.cos(2 * np.pi * R2) * (stallion_pos - group_pop[i])
                        + stallion_pos
                    )

                temp_pos = np.clip(temp_pos, lb, ub)
                new_fit = obj_func(temp_pos)

                if new_fit < group_fit[i]:
                    group_pop[i] = temp_pos
                    group_fit[i] = new_fit
                    if new_fit < best_fit:
                        global_best = temp_pos.copy()
                        best_fit = new_fit

            # Update group
            pop[group_start:group_end] = group_pop
            fitness[group_start:group_end] = group_fit

            # Phase 3: Breeding Activity (Simplified crossover for offspring)
            if np.random.rand() < pc:
                # Select two parents randomly from group
                parent1_idx = np.random.randint(0, group_size)
                parent2_idx = np.random.randint(0, group_size)
                if parent1_idx != parent2_idx:
                    offspring = 0.5 * (
                        group_pop[parent1_idx] + group_pop[parent2_idx]
                    )  # Simple crossover
                    offspring = np.clip(offspring, lb, ub)
                    off_fit = obj_func(offspring)
                    # Replace worst in group if better
                    worst_idx = np.argmax(group_fit)
                    if off_fit < group_fit[worst_idx]:
                        group_pop[worst_idx] = offspring
                        group_fit[worst_idx] = off_fit
                        pop[group_start:group_end] = group_pop
                        fitness[group_start:group_end] = group_fit
                        if off_fit < best_fit:
                            global_best = offspring.copy()
                            best_fit = off_fit

            # Phase 4: Group Leadership (Stallion update towards global best)
            R4 = np.random.rand(dim)
            R6 = np.random.rand()
            if R6 < 0.5:
                new_stallion = (
                    2 * R4 * np.sin(2 * np.pi * R5) * (global_best - stallion_pos)
                    + global_best
                )
            else:
                new_stallion = (
                    2 * R4 * np.cos(2 * np.pi * R5) * (global_best - stallion_pos)
                    + global_best
                )

            new_stallion = np.clip(new_stallion, lb, ub)
            new_stall_fit = obj_func(new_stallion)

            if new_stall_fit < group_fit[stallion_idx]:
                group_pop[stallion_idx] = new_stallion
                group_fit[stallion_idx] = new_stall_fit
                pop[group_start:group_end] = group_pop
                fitness[group_start:group_end] = group_fit
                if new_stall_fit < best_fit:
                    global_best = new_stallion.copy()
                    best_fit = new_stall_fit

        # Phase 5: Leadership Transition (Check if any zebra better than stallion - already handled in feeding)

        convergence[t - 1] = best_fit
        if verbose and t % 50 == 0:
            print(f"Iteration {t}: Best Fitness = {best_fit:.4f}")

    return global_best, best_fit, convergence, pop, fitness


# Objective function: Negative R2 (maximize R2 -> minimize -R2)
def objective(params):
    alpha = params[0]
    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver="saga",
        max_iter=10000,
        tol=0.001,
        random_state=42,
    )
    scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
    return -np.mean(scores)  # Minimize negative mean R2


# Run ZOA for optimization
best_params, best_fit, convergence, _, _ = zebra_optimizer(
    objective, lb, ub, dim, pop_size=30, max_iter=200, verbose=True
)

# Best hyperparameters
best_alpha = best_params[0]
print(f"\nFinal Hyperparameters: alpha={best_alpha:.4f}")
print(f"Best CV R2: {-best_fit:.4f}")

# Plot Convergence
plt.figure(figsize=(10, 6))
plt.plot(convergence, "b-", linewidth=2)
plt.title("ZOA Convergence Curve (200 Iterations)")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (-Mean R2)")
plt.grid(True)
plt.show()

# Train final model with best params
final_model = Ridge(
    alpha=best_alpha,
    fit_intercept=True,
    solver="saga",
    max_iter=10000,
    tol=0.001,
    random_state=42,
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

print("\n📋 Performance Metrics Table (Optimized Ridge):")
print(metrics_df)
