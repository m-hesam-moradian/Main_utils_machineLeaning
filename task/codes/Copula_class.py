import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier  # Classification model
import matplotlib.pyplot as plt
from itertools import combinations


# ----------------------------- #
# 1. Define the sensitivity function
# ----------------------------- #
def couples_sensitivity_analysis(
    model, X, y, feature_pairs, metric="accuracy", perturbation=0.1
):
    if metric == "mse":
        metric_func = mean_squared_error
    elif metric == "mae":
        metric_func = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    elif metric == "accuracy":
        metric_func = accuracy_score
    else:
        raise ValueError("Unsupported metric. Choose from 'mse', 'mae', 'accuracy'.")

    original_predictions = model.predict(X)
    original_score = metric_func(y, original_predictions)

    sensitivity_report = []

    for feature_1, feature_2 in feature_pairs:
        X_perturbed = X.copy()
        X_perturbed[feature_1] *= 1 + perturbation
        X_perturbed[feature_2] *= 1 + perturbation

        perturbed_predictions = model.predict(X_perturbed)
        perturbed_score = metric_func(y, perturbed_predictions)
        sensitivity = perturbed_score - original_score

        sensitivity_report.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "original_score": original_score,
                "perturbed_score": perturbed_score,
                "sensitivity": sensitivity,
            }
        )

    return pd.DataFrame(sensitivity_report)


# ----------------------------- #
# 2. Load data and run analysis
# ----------------------------- #
data_Path = r"D:\ML\Main_utils_machineLeaning\task\BSE. No.13-Dataset.xlsx"
df = pd.read_excel(data_Path, sheet_name="Balanced_Shuffled")

target_column = "Cyberattack_Detected"
X = df.drop(columns=[target_column])
y = df[target_column]

model = ExtraTreesClassifier()
model.fit(X, y)

# Generate feature pairs
features = X.columns
feature_pairs = [
    (features[i], features[j])
    for i in range(len(features))
    for j in range(len(features))
]

# Run sensitivity analysis
copula = couples_sensitivity_analysis(
    model, X, y, feature_pairs, metric="accuracy", perturbation=0.1
)

# ----------------------------- #
# 3. Aggregate copula by feature_1
# ----------------------------- #
g_data = copula.groupby("feature_1")
mean_values_list = []

for v, groupD in g_data:
    numerical_cols = groupD.select_dtypes(include=[int, float])
    mean_values = numerical_cols.mean()
    mean_values_list.append(mean_values.to_frame().T)

copula_average = pd.concat(mean_values_list, ignore_index=True)
copula_average.reset_index(inplace=True)
copula_average.rename(columns={"index": "feature_1"}, inplace=True)

# ----------------------------- #
# 4. Return both results
# ----------------------------- #
# print("\n--- Copula Sensitivity Report ---")
# print(copula)

# print("\n--- Aggregated Copula (Average by feature_1) ---")
# print(copula_average)
with pd.ExcelWriter(
    data_Path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    copula.to_excel(writer, sheet_name="Copula", index=False)
    copula_average.to_excel(writer, sheet_name="Copula_Average", index=False)
