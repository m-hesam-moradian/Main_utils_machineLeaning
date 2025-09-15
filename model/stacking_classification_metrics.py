import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# from metrics.getAllMetric import getAllMetric


# Path & target
DATA_PATH = r"D:\ML\main_structure\data\data.xlsx"
TARGET = "Stress Level "

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="Rnd2_STACKING_3_fold")
print("Columns in DataFrame:", df.columns.tolist())

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define default parameters for XGB
# Define shared parameters
shared_params = {
    "n_estimators": 150,  # Lower number of trees
    "max_depth": 5,  # Shallower trees generalize better
    "random_state": 42,
}


# Define base learners using shared parameters
estimators = [
    (
        "xgb",
        XGBClassifier(
            n_estimators=102,
            
            learning_rate=0,
            random_state=shared_params["random_state"],
        ),
    ),
    (
        "rfc",
        RandomForestClassifier(
            n_estimators=103,
            max_depth=20,
            random_state=shared_params["random_state"],
        ),
    ),
]


# Define stacking model
model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=10003, random_state=42),
)

# Fit model
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


# Capture default parameters
param_names = list(shared_params.keys())
param_values = list(shared_params.values())
default_params_df = pd.DataFrame([param_values], columns=param_names)

# Display results
print("\n--- Default XGBoost Parameters ---")
print(default_params_df)
print("\n--- Performance Metrics ---")
# print(metrics_df)
print("\n--- Train Predictions ---")
print(train_df.head())
print("\n--- Test First Half Predictions ---")
print(test_first_half_df.head())
print("\n--- Test Second Half Predictions ---")
print(test_second_half_df.head())


import numpy as np
import zipfile
import io

# Step 1: Stack all y_true and y_pred pairs
combined = np.concatenate(
    [
        np.column_stack((y_train.to_numpy(), y_train_pred)),
        np.column_stack((y_test_first_half.to_numpy(), y_test_first_pred)),
        np.column_stack((y_test_second_half.to_numpy(), y_test_second_pred)),
    ]
)

pd.DataFrame(combined)
