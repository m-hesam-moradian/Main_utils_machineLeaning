import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from model.metrics.error_classification import getAllMetric


def train_model(X_train, y_train, X_test, y_test, params=None):
    # Model selection (same as your current setup)

    default_params = {
        "n_estimators": 50,
        "learning_rate": 0.5,
        "random_state": 42,
        "loss": "square",
    }

    final_params = (
        default_params
        if params is None
        else {
            "n_estimators": int(params[0]),
            "learning_rate": float(params[1]),
            "loss": "linear",
        }
    )

    # Initialize the model with the final parameters
    model = AdaBoostRegressor(**final_params)
    # Model name
    model_name = "AdaBoostRegressor"
    # Fit the model
    np.asarray(y_train)

    np.asarray(y_train)
    np.asarray(y_test)
    model.fit(X_train, y_train)
    # Predictions

    y_pred_train = model.predict(X_train)
    midpoint = len(y_test) // 2
    y_pred_test = model.predict(X_test)

    X_value, X_value_test = X_test[:midpoint], X_test[midpoint:]
    y_value, y_value_test = y_test[:midpoint], y_test[midpoint:]

    y_pred_value = model.predict(X_value)
    y_pred_value_test = model.predict(X_value_test)

    # Evaluate
    # metrics_train = getAllMetric(y_train, y_pred_train)
    # metrics_test = getAllMetric(y_test, y_pred_test)
    # metrics_value = getAllMetric(y_value, y_pred_value)
    # metrics_value_test = getAllMetric(y_value_test, y_pred_value_test)

    # Concatenate actual and predicted
    # y_all = np.concatenate([y_train, y_test])
    # y_pred_all = np.concatenate([y_pred_train, y_pred_test])

    # metrics_all = getAllMetric(y_all, y_pred_all)

    # Before passing to getAllMetric, ensure both are numpy arrays
    metrics_train = pd.DataFrame([getAllMetric(y_train.to_numpy(), y_pred_train)])
    metrics_test = pd.DataFrame([getAllMetric(y_test.to_numpy(), y_pred_test)])
    metrics_value = pd.DataFrame([getAllMetric(y_value.to_numpy(), y_pred_value)])

    metrics_value_test = pd.DataFrame(
        [getAllMetric(y_value_test.to_numpy(), y_pred_value_test)]
    )

    # For concatenated all predictions
    metrics_all = pd.DataFrame(
        [
            getAllMetric(
                np.concatenate([y_train.to_numpy(), y_test.to_numpy()]),
                np.concatenate([y_pred_train, y_pred_test]),
            )
        ]
    )

    # Bundle all metrics
    all_metrics = {
        "all": metrics_all,
        "train": metrics_train,
        "test": metrics_test,
        "value": metrics_value,
        "test_value": metrics_value_test,
    }

    # Build the result dictionary
    return {
        "model_name": model_name,
        "best_params": pd.DataFrame([final_params]),
        "metrics": all_metrics,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
    }

    # # Add the model only if params is not None
    # if params is not None:
    #     result["model"] = model

    # return result
