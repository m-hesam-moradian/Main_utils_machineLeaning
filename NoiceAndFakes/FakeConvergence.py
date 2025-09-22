import pandas as pd
import numpy as np


def get_conv(count=200, low=0.08, high=0.22, minPhase=6, maxPhase=10, cov="rmse"):
    # Generate a random phase between minPhase and maxPhase
    phase = np.random.randint(minPhase, maxPhase + 1)

    convergence = []

    for _ in range(phase):
        repeated_count = np.random.randint(1, 6)  # Adjust the range as needed
        random_number = np.random.uniform(low, high)
        repeated_numbers = [random_number] * repeated_count

        # Extend the convergence array with the specified values
        convergence.extend(repeated_numbers)

    # Trim or repeat values to match the specified count
    convergence = np.resize(convergence, count)
    if cov == "rmse":
        # Sort the array from high to low
        convergence = np.sort(convergence)[::-1]
    else:
        convergence = np.sort(convergence)[::1]

    # Ensure the lowest repeated number is equal to the specified low value
    # convergence[-1] = low

    return np.array(convergence)


convergence_rmse = get_conv(
    count=200, high=9.74251
, low=2.504427011





, minPhase=24, maxPhase=32, cov="rmse"
)
