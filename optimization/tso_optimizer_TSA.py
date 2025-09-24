import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from Metrics_regression import (
    getAllMetric,
)  # Assuming this is your custom metrics function

# Load data
sheet_name = "Data after K-Fold (GBR & ANFIS)"
df = pd.read_excel(
    r"D:\ML\Main_utils\Task\Global_AI_Content_Impact_Dataset.xlsx",
    sheet_name=sheet_name,
)

# Replace NULL values with 0
df = df.fillna(0)

target_column = "Market Share of AI Companies (%)"

# Features and Target
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


# Transient Search Optimization (TSO) Function
def tso_optimize(objective, dim, lb, ub, pop_size=30, max_iter=50):
    """
    Transient Search Optimization Algorithm for hyperparameter tuning.
    Objective: maximization (e.g., return -loss for minimization).
    """
    # Initialize population
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([objective(ind) for ind in pop])

    # Find initial best
    best_idx = np.argmax(fitness)
    X_best = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)  # Update coefficient a

        for i in range(pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            # Position update
            pop[i] = X_best + a * np.exp(-r1 * t) * (r2 * pop[i] - X_best)
            # Boundary enforcement
            pop[i] = np.clip(pop[i], lb, ub)

        # Evaluate new population
        fitness = np.array([objective(ind) for ind in pop])

        # Update best
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fit:
            best_fit = fitness[best_idx]
            X_best = pop[best_idx].copy()

    return X_best, best_fit


# Objective functions (minimize RMSE, so return -RMSE)
def objective_adar(params):
    n_estimators = int(params[0])
    learning_rate = params[1]
    try:
        model = AdaBoostRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        _, rmse, _, _, _ = getAllMetric(y_test, y_pred)
        return -rmse  # Negative for maximization
    except:
        return -np.inf  # Penalty for invalid params


def objective_catr(params):
    iterations = int(params[0])
    learning_rate = params[1]
    depth = int(params[2])
    l2_leaf_reg = params[3]
    try:
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=42,
            verbose=0,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        _, rmse, _, _, _ = getAllMetric(y_test, y_pred)
        return -rmse  # Negative for maximization
    except:
        return -np.inf  # Penalty for invalid params


# Optimize ADAR
print("Optimizing AdaBoostRegressor (ADAR)...")
lb_adar = np.array([50, 0.01])
ub_adar = np.array([500, 1.0])
dim_adar = 2
best_params_adar, best_score_adar = tso_optimize(
    objective_adar, dim_adar, lb_adar, ub_adar
)
print(
    f"Best ADAR params: n_estimators={int(best_params_adar[0])}, learning_rate={best_params_adar[1]:.4f}"
)
print(f"Best ADAR score (neg RMSE): {best_score_adar:.4f}")

# Train final ADAR model and get metrics
model_adar = AdaBoostRegressor(
    n_estimators=int(best_params_adar[0]),
    learning_rate=best_params_adar[1],
    random_state=42,
)
model_adar.fit(X_train, y_train)
y_pred_adar_all = model_adar.predict(X)
y_pred_adar_train = model_adar.predict(X_train)
y_pred_adar_test = model_adar.predict(X_test)

# Split test for ADAR
mid_index = len(y_pred_adar_test) // 2
y_test_first_half = y_test.iloc[:mid_index]
y_test_second_half = y_test.iloc[mid_index:]
y_pred_adar_test_first_half = y_pred_adar_test[:mid_index]
y_pred_adar_test_second_half = y_pred_adar_test[mid_index:]

# Metrics for ADAR
metrics_data_adar = {"Set": [], "R2": [], "RMSE": [], "MAE": [], "RSE": [], "SMAPE": []}
sets_adar = [
    ("All", y, y_pred_adar_all),
    ("Train", y_train, y_pred_adar_train),
    ("Test", y_test, y_pred_adar_test),
    ("Value", y_test_first_half, y_pred_adar_test_first_half),
    ("Test-Value", y_test_second_half, y_pred_adar_test_second_half),
]
for name, y_true, y_pred in sets_adar:
    R, RMSE, MAE, RSE, SMAPE = getAllMetric(y_true, y_pred)
    metrics_data_adar["Set"].append(name)
    metrics_data_adar["R2"].append(R)
    metrics_data_adar["RMSE"].append(RMSE)
    metrics_data_adar["MAE"].append(MAE)
    metrics_data_adar["RSE"].append(RSE)
    metrics_data_adar["SMAPE"].append(SMAPE)
metrics_df_adar = pd.DataFrame(metrics_data_adar)
print("\n📋 ADAR Performance Metrics Table:")
print(metrics_df_adar)

# Optimize CATR
print("\nOptimizing CatBoostRegressor (CATR)...")
lb_catr = np.array([500, 0.00, 4, 1.0])
ub_catr = np.array([2000, 0.01, 10, 10.0])
dim_catr = 4
best_params_catr, best_score_catr = tso_optimize(
    objective_catr, dim_catr, lb_catr, ub_catr
)
print(
    f"Best CATR params: iterations={int(best_params_catr[0])}, learning_rate={best_params_catr[1]:.4f}, depth={int(best_params_catr[2])}, l2_leaf_reg={best_params_catr[3]:.4f}"
)
print(f"Best CATR score (neg RMSE): {best_score_catr:.4f}")

# Train final CATR model and get metrics
model_catr = CatBoostRegressor(
    iterations=int(best_params_catr[0]),
    learning_rate=best_params_catr[1],
    depth=int(best_params_catr[2]),
    l2_leaf_reg=best_params_catr[3],
    random_seed=42,
    verbose=0,
)
model_catr.fit(X_train, y_train)
y_pred_catr_all = model_catr.predict(X)
y_pred_catr_train = model_catr.predict(X_train)
y_pred_catr_test = model_catr.predict(X_test)

# Split test for CATR
y_pred_catr_test_first_half = y_pred_catr_test[:mid_index]
y_pred_catr_test_second_half = y_pred_catr_test[mid_index:]

# Metrics for CATR
metrics_data_catr = {"Set": [], "R2": [], "RMSE": [], "MAE": [], "RSE": [], "SMAPE": []}
sets_catr = [
    ("All", y, y_pred_catr_all),
    ("Train", y_train, y_pred_catr_train),
    ("Test", y_test, y_pred_catr_test),
    ("Value", y_test_first_half, y_pred_catr_test_first_half),
    ("Test-Value", y_test_second_half, y_pred_catr_test_second_half),
]
for name, y_true, y_pred in sets_catr:
    R, RMSE, MAE, RSE, SMAPE = getAllMetric(y_true, y_pred)
    metrics_data_catr["Set"].append(name)
    metrics_data_catr["R2"].append(R)
    metrics_data_catr["RMSE"].append(RMSE)
    metrics_data_catr["MAE"].append(MAE)
    metrics_data_catr["RSE"].append(RSE)
    metrics_data_catr["SMAPE"].append(SMAPE)
metrics_df_catr = pd.DataFrame(metrics_data_catr)
print("\n📋 CATR Performance Metrics Table:")
print(metrics_df_catr)
