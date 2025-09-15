from src.model.train_model import train_raw_model
import numpy as np
import pandas as pd


# 🗂️ Storage for all optimizer results
optimizer_results = {}


def get_optimizer_report(
    name,
    model_type,
    best_params,
    best_fit,
    X_train,
    y_train,
    X_test,
    y_test,
    convergence=None,
):
    # Combine train and test sets
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    # Train model with best parameters
    result = train_raw_model(
        model_type, X_train, y_train, X_test, y_test, params=best_params
    )

    # Make predictions on both sets
    y_pred_train = result["model"].predict(X_train)
    y_pred_test = result["model"].predict(X_test)
    y_pred_all = np.concatenate([y_pred_train, y_pred_test])

    return {
        "real_y": y_all.to_numpy(),
        "prediction": y_pred_all,
        "best_params": best_params,
        "best_mae": float(best_fit),
        "convergence": np.array(convergence),
        "metrics": result["metrics"],
        "model_name": result["model_name"],
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
