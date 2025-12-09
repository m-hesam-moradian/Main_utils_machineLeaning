import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score


def couples_sensitivity_analysis(
    model, X, y, feature_pairs, metric="mse", perturbation=0.1
):
    """
    Perform sensitivity analysis for couples (pairs) of features in a machine learning model.

    Parameters:
    -----------
    model : trained machine learning model
        The trained model object with a `predict` method.

    X : pd.DataFrame or np.ndarray
        Input data for the model.

    y : pd.Series or np.ndarray
        True labels or target values.

    feature_pairs : list of tuples
        List of pairs of feature names or indices for sensitivity analysis.

    metric : str, optional, default='mse'
        Evaluation metric to measure the change in predictions.
        Options: 'mse', 'mae', 'accuracy'.

    perturbation : float, optional, default=0.1
        The percentage by which to perturb the feature values.

    Returns:
    --------
    pd.DataFrame
        Sensitivity report for each feature pair, showing the effect of perturbations.
    """

    # Define metric function
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
        # Copy the data for perturbation
        X_perturbed = X.copy()

        # Perturb both features
        if isinstance(X_perturbed, pd.DataFrame):
            X_perturbed[feature_1] *= 1 + perturbation
            X_perturbed[feature_2] *= 1 + perturbation
        else:
            X_perturbed[:, feature_1] *= 1 + perturbation
            X_perturbed[:, feature_2] *= 1 + perturbation

        # Get new predictions
        perturbed_predictions = model.predict(X_perturbed)

        # Calculate score after perturbation
        perturbed_score = metric_func(y, perturbed_predictions)

        # Calculate sensitivity as the difference in metric
        sensitivity = perturbed_score - original_score

        # Store the results
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
