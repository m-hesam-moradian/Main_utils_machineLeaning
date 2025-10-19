import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

# --- Load data ---
sheet_name = "Data_after_KFold"
excel_path = r"D:\ML\Main_utils_machineLeaning\task\BSE. No.14-Dataset.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Anomaly_Detected"

# --- Encode Target Variable ---
# Binary encoding based on median if numeric
if np.issubdtype(df[target_column].dtype, np.number):
    median_value = df[target_column].median()
    y = (df[target_column] > median_value).astype(int)
else:
    y = df[target_column].astype("category").cat.codes

# --- Preprocess Features ---
categorical_cols = df.select_dtypes(include=["object"]).columns.drop(
    target_column, errors="ignore"
)
X = pd.get_dummies(
    df.drop(columns=[target_column]), columns=categorical_cols, drop_first=True
)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False, random_state=42
)

# --- SGDC Model ---


model = SGDClassifier(
   loss="log_loss",             # Logistic regression loss
    penalty="elasticnet",        # Combines L1 and L2 regularization
    alpha=0.001,                 # Stronger regularization
    l1_ratio=0.7,                # More weight on L1 (sparsity)
    fit_intercept=True,
    max_iter=3000,               # Reasonable iteration count
    tol=1e-4,                    # Tighter convergence tolerance
    shuffle=True,
    verbose=0,
    epsilon=0.1,
    n_jobs=None,
    random_state=42,
    learning_rate="adaptive",    # Adjusts learning rate based on performance
    eta0=0.01,                   # Initial learning rate
    power_t=0.5,
    early_stopping=True,        # Stop if no improvement
    validation_fraction=0.2,    # Larger validation set for early stopping
    n_iter_no_change=10,
    class_weight="balanced",    # Adjusts for class imbalance
    warm_start=False,
    average=True                  # Whether to use averaged SGD weights
)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Split Test Predictions ---
mid_index = len(y_pred_test) // 2
y_test_first_half = y_test[:mid_index]
y_test_second_half = y_test[mid_index:]
y_pred_test_first_half = y_pred_test[:mid_index]
y_pred_test_second_half = y_pred_test[mid_index:]


# --- Build Metrics Table ---
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }


metrics_data = {
    "Set": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "F2": [],
}

sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test_first_half, y_pred_test_first_half),
    ("Test-Value", y_test_second_half, y_pred_test_second_half),
]

for name, y_true, y_pred in sets:
    metrics = get_metrics(y_true, y_pred)
    metrics_data["Set"].append(name)
    for key in metrics:
        metrics_data[key].append(metrics[key])

metrics_df = pd.DataFrame(metrics_data)

# --- Create DataFrames for real vs predicted ---
df_train = pd.DataFrame({"y_train_real": y_train, "y_train_pred": y_pred_train})
df_test = pd.DataFrame({"y_test_real": y_test, "y_test_pred": y_pred_test})
df_all = pd.concat(
    [
        pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train}),
        pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test}),
    ],
    ignore_index=True,
)

# --- Print Metrics Table ---
print("\n📋 Performance Metrics Table (SGDC):")
print(metrics_df)
