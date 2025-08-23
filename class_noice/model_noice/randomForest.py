import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Function to generate fake predictions with a given accuracy ---
def generate_fake_predictions(y_true, desired_accuracy, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    y_true = np.array(y_true)
    n_samples = len(y_true)
    n_correct = int(desired_accuracy * n_samples)
    n_wrong = n_samples - n_correct

    # possible labels
    unique_labels = np.unique(y_true)

    # random indices for correct/wrong predictions
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    correct_indices = indices[:n_correct]
    wrong_indices = indices[n_correct:]

    # build y_pred
    y_pred = np.empty_like(y_true)
    y_pred[correct_indices] = y_true[correct_indices]

    # wrong predictions
    for idx in wrong_indices:
        wrong_choices = unique_labels[unique_labels != y_true[idx]]
        y_pred[idx] = np.random.choice(wrong_choices)

    return y_pred

# --- Path & target ---
DATA_PATH = r"D:\ML\Revise 566-Tasks for Code\data\data.xlsx"
TARGET = "attack"

# --- Load dataset ---
df = pd.read_excel(DATA_PATH, sheet_name="data")
X = df.drop(columns=[TARGET])

# Encode categorical features
categorical_cols = ["protocol_type", "service", "flag"]
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encode target
le = LabelEncoder()
y = le.fit_transform(df[TARGET])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Random Forest Classifier ---
default_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
}
model = RandomForestClassifier(**default_params)

# --- Train model ---
start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

start_time = time.time()
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
predict_time = time.time() - start_time

# Split test into two halves
half = len(X_test) // 2
y_test_first_half = y_test[:half]
y_test_first_pred = y_test_pred[:half]
y_test_second_half = y_test[half:]
y_test_second_pred = y_test_pred[half:]

# DataFrames for y_true and y_pred
train_df = pd.DataFrame({"y_true": y_train, "y_pred": y_train_pred})
test_first_half_df = pd.DataFrame({"y_true": y_test_first_half, "y_pred": y_test_first_pred})
test_second_half_df = pd.DataFrame({"y_true": y_test_second_half, "y_pred": y_test_second_pred})

# --- Metrics function ---
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-score": f1_score(y_true, y_pred, average="weighted"),
    }

# --- Metrics for real model predictions ---
metrics_dict = {
    "All": compute_metrics(y, model.predict(X)),
    "Train": compute_metrics(y_train, y_train_pred),
    "Test": compute_metrics(y_test, y_test_pred),
    "Value": compute_metrics(y_test_first_half, y_test_first_pred),
    "Test_Half": compute_metrics(y_test_second_half, y_test_second_pred),
}
metrics_df = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={"index": "Dataset"})

# --- Generate fake/noisy predictions ---
desired_accuracy = 0.85

# Fake predictions for train and test separately
y_train_fake = generate_fake_predictions(y_train, desired_accuracy, random_seed=42)
y_test_fake = generate_fake_predictions(y_test, desired_accuracy, random_seed=42)

# Split test into halves
half = len(y_test) // 2
y_test_first_fake = y_test_fake[:half]
y_test_second_fake = y_test_fake[half:]

# Compute metrics
metrics_fake_dict = {
    "All": compute_metrics(np.concatenate([y_train, y_test]), np.concatenate([y_train_fake, y_test_fake])),
    "Train": compute_metrics(y_train, y_train_fake),
    "Test": compute_metrics(y_test, y_test_fake),
    "Value": compute_metrics(y_test[:half], y_test_first_fake),
    "Test_Half": compute_metrics(y_test[half:], y_test_second_fake),
}

metrics_fake_df = pd.DataFrame(metrics_fake_dict).T.reset_index().rename(columns={"index": "Dataset"})


# --- Runtime report ---
runtime_df = pd.DataFrame({"Task": ["Training", "Prediction"], "Time_seconds": [train_time, predict_time]})
params_df = pd.DataFrame([default_params])

# --- Display ---
print("\n--- Random Forest Parameters ---")
print(params_df)

print("\n--- Performance Metrics (Model) ---")
print(metrics_df)

print("\n--- Performance Metrics (Fake Predictions) ---")
print(metrics_fake_df)

print("\n--- Runtime Report ---")
print(runtime_df)

print("\n--- Training Predictions Sample ---")
print(train_df.head())

print("\n--- Train DF Length ---")
print(train_df.shape[0])
