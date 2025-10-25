# D:\ML\Project-2\src\model\xgb_classification_metrics.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Path & target
DATA_PATH = r"D:\ML\main_structure\data\data.xlsx"
TARGET = "HVAC Efficiency"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="data_after_balancing")

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define model (without specifying any parameters to use defaults)
default_params = {
    "n_estimators": 100,
    "learning_rate": 2,
    "max_depth": 6,
    "subsample": 1,
    "colsample_bytree": 1,
}


model = XGBClassifier(
    n_estimators=default_params["n_estimators"],
    learning_rate=default_params["learning_rate"],
    max_depth=default_params["max_depth"],
    subsample=default_params["subsample"],
    colsample_bytree=default_params["colsample_bytree"],
    use_label_encoder=False,
    eval_metric="logloss",
)

model.fit(X_train, y_train)

# Predict for train and test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Split test into two halves
half = len(X_test) // 2
y_test_first_half = y_test.iloc[:half]
y_test_first_pred = y_test_pred[:half]
y_test_second_half = y_test.iloc[half:]
y_test_second_pred = y_test_pred[half:]

# Create DataFrames for y_true and y_pred
train_df = pd.DataFrame({"y_true": y_train, "y_pred": y_train_pred})
test_first_half_df = pd.DataFrame(
    {"y_true": y_test_first_half, "y_pred": y_test_first_pred}
)
test_second_half_df = pd.DataFrame(
    {"y_true": y_test_second_half, "y_pred": y_test_second_pred}
)


# Performance metrics function
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-score": f1_score(y_true, y_pred, average="weighted"),
    }


# Metrics for 5 parts
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

# Create DataFrame of default model parameters (one row)


# Catch the parameter values into a DataFrame
param_names = list(default_params.keys())
param_values = list(default_params.values())

default_params_df = pd.DataFrame([param_values], columns=param_names)

print("\n--- Default XGBoost Parameters ---")
print(default_params_df)

# Display results
print("\n--- Default XGBoost Parameters ---")
print(default_params_df)
print("\n--- Performance Metrics ---")
print(metrics_df)
print("\n--- Train Predictions ---")
print(train_df.head())
print("\n--- Test First Half Predictions ---")
print(test_first_half_df.head())
print("\n--- Test Second Half Predictions ---")
print(test_second_half_df.head())
