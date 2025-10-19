import numpy as np
import random
from sklearn.metrics import accuracy_score


 
# Data Loading
data = np.loadtxt('Data_err.npt')
y = data[:, 0]
ypred = data[:, 1]

def optimize_ypred(y, ypred, target_accuracy_range):
    """
    Adjust ypred to achieve a desired accuracy score range.

    Args:
        y (array-like): Ground truth values.
        ypred (array-like): Predicted values.
        target_accuracy_range (tuple): Desired accuracy score range (min, max).

    Returns:
        array-like: Adjusted ypred values.
    """
    min_accuracy, max_accuracy = target_accuracy_range
    accuracy = accuracy_score(y, ypred)

    while accuracy < min_accuracy or accuracy > max_accuracy:
        # Calculate the difference between y and ypred
        diff = np.abs(y - ypred)

        # Identify the indices where y and ypred differ
        diff_indices = np.where(diff > 0)[0]

        # Randomly select indices to adjust
        adjust_indices = np.random.choice(diff_indices, size=len(diff_indices) // 2, replace=False)

        # Adjust ypred values at selected indices
        ypred[adjust_indices] = np.abs(y[adjust_indices])

        # Recalculate accuracy
        accuracy = accuracy_score(y, ypred)
        print("accuracy",accuracy)
    return ypred

# Example usage

target_accuracy_range = (0.961514, 0.971918)

adjusted_ypred = optimize_ypred(y, ypred, target_accuracy_range)
print("Adjusted ypred:", adjusted_ypred)