from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error


def objective_adaboost(params, X_train, y_train, X_test, y_test):
    """
    Objective function for AdaBoostRegressor optimization.

    Parameters:
        params[0] -> n_estimators (int)
        params[1] -> learning_rate (float)
    """
    # Extract and convert parameters
    n_estimators = int(params[0])
    learning_rate = float(params[1])

    # Define model
    model = AdaBoostRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
    )

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    return mae
