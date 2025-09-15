import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Fitness function: negative F1-score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# Updated fitness function
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from xgboost import XGBClassifier


# Logistic chaotic initialization
def logistic_initialization(pop, dim, lb, ub):
    Positions = np.zeros((pop, dim))
    for i in range(pop):
        x0 = np.random.rand()
        a = 4
        for j in range(dim):
            x0 = a * x0 * (1 - x0)
            Positions[i, j] = lb[j] + (ub[j] - lb[j]) * x0
    return Positions


# Levy flight
def levy(dim, beta=1.5):
    sigma = (
        math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / np.abs(v) ** (1 / beta)


def fitness_function(params, X_train, y_train, X_test, y_test):
    rf_n = int(params[0])  # RandomForest n_estimators
    rf_depth = int(params[1])  # RandomForest max_depth
    xgb_n = int(params[2])  # XGBoost n_estimators
    xgb_lr = 10 ** params[3]  # XGBoost learning_rate
    knn_k = int(params[4])  # KNN n_neighbors
    meta_C = 10 ** params[5]  # LogisticRegression C

    estimators = [
        ("rf", RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth)),
        (
            "xgb",
            XGBClassifier(
                n_estimators=xgb_n,
                learning_rate=xgb_lr,
            ),
        ),
    ]

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=meta_C, max_iter=500),
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return -f1_score(y_test, y_pred, average="weighted")


# Main HEOA loop
def HEOA_classification(N, Max_iter, dim, lb, ub, X_train, y_train, X_test, y_test):
    jump_factor = abs(ub[0] - lb[0]) / 1000
    LN = int(N * 0.4)
    EN = int(N * 0.4)
    FN = int(N * 0.1)

    X = logistic_initialization(N, dim, lb, ub)
    fitness = np.array(
        [fitness_function(ind, X_train, y_train, X_test, y_test) for ind in X]
    )
    GBestF = np.min(fitness)
    GBestX = X[np.argmin(fitness)]

    curve = []
    metrics_history = []

    for i in range(Max_iter):
        X_new = np.copy(X)
        for j in range(N):
            if i <= Max_iter / 4:
                X_new[j] = (
                    GBestX * (1 - i / Max_iter)
                    + (np.mean(X[j]) - GBestX)
                    * np.floor(np.random.rand() / jump_factor)
                    * jump_factor
                    + 0.2 * (1 - i / Max_iter) * (X[j] - GBestX) * levy(dim)
                )
            else:
                if j < LN:
                    if np.random.rand() < 0.6:
                        X_new[j] = (
                            0.2
                            * np.cos(np.pi / 2 * (1 - i / Max_iter))
                            * X[j]
                            * np.exp(
                                (-i * np.random.randn()) / (np.random.rand() * Max_iter)
                            )
                        )
                    else:
                        X_new[j] = 0.2 * np.cos(np.pi / 2 * (1 - i / Max_iter)) * X[
                            j
                        ] + np.random.randn(dim)
                elif j < LN + EN:
                    X_new[j] = np.random.randn(dim) * np.exp(
                        (X[-1] - X[j]) / (j + 1) ** 2
                    )
                elif j < LN + EN + FN:
                    X_new[j] = X[j] + 0.2 * np.cos(
                        np.pi / 2 * (1 - i / Max_iter)
                    ) * np.random.rand(dim) * (X[0] - X[j])
                else:
                    X_new[j] = GBestX + (GBestX - X[j]) * np.random.randn()

            # Boundary control
            X_new[j] = np.clip(X_new[j], lb, ub)

        fitness_new = np.array(
            [fitness_function(ind, X_train, y_train, X_test, y_test) for ind in X_new]
        )
        if np.min(fitness_new) < GBestF:
            GBestF = np.min(fitness_new)
            GBestX = X_new[np.argmin(fitness_new)]

        X = X_new
        fitness = fitness_new
        curve.append(GBestF)

        # Track metrics
        C = 10 ** GBestX[0]
        clf = LogisticRegression(C=C, max_iter=500)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics_history.append(
            {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
            }
        )

        print(
            f"Iter {i+1}: F1={-GBestF:.4f}, Acc={metrics_history[-1]['accuracy']:.4f}"
        )

    return GBestX, -GBestF, curve, metrics_history


import pandas as pd


# # Load dataset
# Path & target
DATA_PATH = r"D:\ML\main_structure\data\data.xlsx"
TARGET = "Stress Level "
# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="Rnd2_STACKING_3_fold")
# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Optimization settings
N = 5
Max_iter = 10
dim = 6
lb = np.array(
    [10, 2, 10, -3, 1, -3]
)  # [rf_n, rf_depth, xgb_n, log10(xgb_lr), knn_k, log10(meta_C)]
ub = np.array([200, 20, 200, 0, 30, 3])
# Run the optimizer
best_params, best_score, curve, metrics = HEOA_classification(
    N, Max_iter, dim, lb, ub, X_train, y_train, X_test, y_test
)
print("\nBest Parameters:", best_params)
print("Best F1-score:", best_score)
# Plot convergence
plt.plot(curve)
plt.title("HEOA Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best F1-score")
plt.grid(True)
plt.show()
