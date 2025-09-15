import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# --- Load Data ---
DATA_PATH = r"D:\ML\main_structure\data\data.xlsx"
TARGET = "HVAC Efficiency"

df = pd.read_excel(DATA_PATH, sheet_name="data_after_balancing")
X = df.drop(columns=[TARGET])
y = df[TARGET]

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Performance Metrics Function ---
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-score": f1_score(y_true, y_pred, average="weighted"),
    }


# --- Perfume Optimization Algorithm for XGBoost ---
def perfume_optimization(dim, n_agents=5, max_iter=10):
    # Boundaries for each parameter
    bounds = np.array(
        [
            [50, 500],  # n_estimators
            [0.01, 0.3],  # learning_rate
            [3, 15],  # max_depth
            [0.5, 1.0],  # subsample
            [0.5, 1.0],  # colsample_bytree
        ]
    )

    # Initialize agents
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_agents, dim))
    best_fitness = np.inf
    best_position = None

    for t in range(max_iter):
        alpha = 1 - t / max_iter
        for i in range(n_agents):
            j = np.random.randint(n_agents)
            positions[i] = positions[i] + alpha * np.random.rand(dim) * (
                positions[j] - positions[i]
            )
            # Clip to bounds
            positions[i] = np.clip(positions[i], bounds[:, 0], bounds[:, 1])

            # Create model with current agent's parameters
            params = {
                "n_estimators": int(positions[i, 0]),
                "learning_rate": float(positions[i, 1]),
                "max_depth": int(positions[i, 2]),
                "subsample": float(positions[i, 3]),
                "colsample_bytree": float(positions[i, 4]),
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            fitness = 1 - accuracy_score(y_train, y_pred)  # minimize error

            if fitness < best_fitness:
                best_fitness = fitness
                best_position = positions[i].copy()

    # Return best parameters
    return {
        "n_estimators": int(best_position[0]),
        "learning_rate": float(best_position[1]),
        "max_depth": int(best_position[2]),
        "subsample": float(best_position[3]),
        "colsample_bytree": float(best_position[4]),
    }


# --- Optimize Hyperparameters ---
best_params = perfume_optimization(dim=5, n_agents=15, max_iter=30)
print("\n--- Optimized XGBoost Parameters ---")
print(best_params)

# --- Train Optimized Model ---
model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# --- Predictions ---
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
half = len(X_test) // 2
y_test_first_half = y_test.iloc[:half]
y_test_first_pred = y_test_pred[:half]
y_test_second_half = y_test.iloc[half:]
y_test_second_pred = y_test_pred[half:]

train_df = pd.DataFrame({"y_true": y_train, "y_pred": y_train_pred})
test_first_half_df = pd.DataFrame(
    {"y_true": y_test_first_half, "y_pred": y_test_first_pred}
)
test_second_half_df = pd.DataFrame(
    {"y_true": y_test_second_half, "y_pred": y_test_second_pred}
)

# --- Compute Metrics ---
all_metrics = compute_metrics(y, model.predict(X))
train_metrics = compute_metrics(y_train, y_train_pred)
test_metrics = compute_metrics(y_test, y_test_pred)
value_metrics = compute_metrics(y_test_first_half, y_test_first_pred)
test_half_metrics = compute_metrics(y_test_second_half, y_test_second_pred)

metrics_df = pd.DataFrame(
    [
        {"Dataset": "All", **all_metrics},
        {"Dataset": "Train", **train_metrics},
        {"Dataset": "Test", **test_metrics},
        {"Dataset": "Value", **value_metrics},
        {"Dataset": "Test_Half", **test_half_metrics},
    ]
)

# --- Parameter DataFrame ---
param_df = pd.DataFrame([best_params])

# --- Display Results ---
print("\n--- Optimized XGBoost Parameters ---")
print(param_df)
print("\n--- Performance Metrics ---")
print(metrics_df)
print("\n--- Train Predictions ---")
print(train_df.head())
print("\n--- Test First Half Predictions ---")
print(test_first_half_df.head())
print("\n--- Test Second Half Predictions ---")
print(test_second_half_df.head())
