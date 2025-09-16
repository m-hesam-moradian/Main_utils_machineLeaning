import numpy as np
import pandas as pd
import math
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# --- Logistic Chaotic Initialization ---
def logistic_initialization(pop, dim, lb, ub):
    Positions = np.zeros((pop, dim))
    for i in range(pop):
        x0 = np.random.rand()
        a = 4
        for j in range(dim):
            x0 = a * x0 * (1 - x0)
            Positions[i, j] = lb[j] + (ub[j] - lb[j]) * x0
    return Positions


# --- Fitness Evaluation ---
def evaluate_metrics(params, X_train, y_train, X_test, y_test):
    rf_n = int(params[0])
    rf_depth = int(params[1])
    xgb_n = int(params[2])
    xgb_lr = 10 ** params[3]
    knn_k = int(params[4])
    meta_C = 10 ** params[5]

    estimators = [
        ("rf", RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth)),
        ("xgb", XGBClassifier(n_estimators=xgb_n, learning_rate=xgb_lr)),
        ("knn", KNeighborsClassifier(n_neighbors=knn_k)),
    ]

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=meta_C, max_iter=500),
        passthrough=True,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "params": params,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


# --- Load Dataset ---
DATA_PATH = r"D:\ML\main_structure\data\data.xlsx"
TARGET = "Stress Level "
df = pd.read_excel(DATA_PATH, sheet_name="Rnd2_STACKING_3_fold")
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- APO Settings ---
N = 10  # Number of hyperparameter sets
dim = 6
lb = np.array([10, 2, 10, -3, 1, -3])  # Lower bounds
ub = np.array([200, 20, 200, 0, 30, 3])  # Upper bounds

# --- Generate and Evaluate ---
param_sets = logistic_initialization(N, dim, lb, ub)
results = [
    evaluate_metrics(params, X_train, y_train, X_test, y_test) for params in param_sets
]

# --- Display Results ---
metrics_df = pd.DataFrame(
    [
        {
            "Set": f"Config {i+1}",
            "RF_n": int(r["params"][0]),
            "RF_depth": int(r["params"][1]),
            "XGB_n": int(r["params"][2]),
            "XGB_lr": round(10 ** r["params"][3], 5),
            "KNN_k": int(r["params"][4]),
            "Meta_C": round(10 ** r["params"][5], 5),
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1": round(r["f1"], 4),
        }
        for i, r in enumerate(results)
    ]
)

print("\n📊 APO Hyperparameter Evaluation Table:")
print(metrics_df)
