import time
import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze

start_time = time.time()


def Morris_function(X, predictions):

    column_names = X.columns

    # Convert to float arrays
    X = np.asarray(X, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)

    D = X.shape[1]
    trajectory_size = D + 1

    problem = {
        "num_vars": D,
        "names": list(column_names),
        "bounds": [
            [float(np.min(X[:, i])), float(np.max(X[:, i]))]
            for i in range(D)
        ],
    }

    # -------------------------------------------------------
    # Align lengths
    # -------------------------------------------------------

    N_x = len(X)
    N_y = len(predictions)

    min_length = min(N_x, N_y)
    valid_len = (min_length // trajectory_size) * trajectory_size

    print(f"Original X length: {N_x}")
    print(f"Original Predictions length: {N_y}")
    print(
        f"Using {valid_len} samples "
        f"(multiple of trajectory size = {trajectory_size})"
    )

    X = X[:valid_len]
    predictions = predictions[:valid_len]

    # Ensure float dtype
    X = X.astype(np.float64)
    predictions = predictions.astype(np.float64)

    # -------------------------------------------------------
    # Morris Analysis
    # -------------------------------------------------------

    start_analysis = time.time()

    Si = morris_analyze.analyze(
        problem,
        X,
        predictions,
        num_levels=4,
        print_to_console=False,
    )

    end_analysis = time.time()

    results = pd.DataFrame({
        "parameter": problem["names"],
        "mu": Si["mu"],
        "mu_star": Si["mu_star"],
        "sigma": Si["sigma"],
        "mu_star_conf": Si["mu_star_conf"],
    })

    print(
        f"Morris completed in "
        f"{end_analysis - start_analysis:.2f} seconds."
    )

    return results


# ==========================================================
# Load Data
# ==========================================================

DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

df = pd.read_excel(
    DATA_PATH,
    sheet_name="Balanced_SMOTE_ENN"
).dropna()

target_column = df.columns[-1]

X = df.drop(columns=[target_column])

# ==========================================================
# Load Predictions
# ==========================================================

y = (
    pd.read_csv(
        r"C:\Users\Sam\Desktop\ML\data\predictions.txt",
        header=None
    )
    .iloc[:, 0]
    .astype(np.float64)
    .to_numpy()
)

print("Features Shape:", X.shape)
print("Predictions Shape:", y.shape)
print("Prediction dtype:", y.dtype)

# ==========================================================
# Run Morris
# ==========================================================

Morris_df = Morris_function(
    X=X,
    predictions=y
)

print("\nFinal Results:")
print(Morris_df)

Morris_df.to_clipboard(index=False)

print("\nResults successfully copied to clipboard!")