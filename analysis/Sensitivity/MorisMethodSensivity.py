import time
import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze

start_time = time.time()


def Morris_function(X, predictions):
    column_names = X.columns
    X = np.array(X)
    predictions = np.array(predictions)
    """ 
    Inputs: 
      - X: Array or DataFrame of shape (N_samples × D_features) 
      - predictions: Model outputs as a 1D array
    Output: 
      - DataFrame containing mu, mu_star, sigma, and confidence intervals for each parameter 
    """

    D = X.shape[1]
    trajectory_size = D + 1  # Morris requires exactly D + 1 samples per trajectory

    problem = {
        "num_vars": D,
        "names": list(column_names),
        "bounds": [[np.min(X[:, i]), np.max(X[:, i])] for i in range(D)],
    }

    # 1) Force alignment to match Morris Trajectory constraints (D + 1)
    N_x = len(X)
    N_y = len(predictions)
    min_length = min(N_x, N_y)

    # Find the nearest perfect multiple of (D + 1)
    valid_len = (min_length // trajectory_size) * trajectory_size

    print(f"Original X length: {N_x}, Predictions length: {N_y}")
    print(
        f"Trimming data to length {valid_len} (nearest multiple of trajectory size {trajectory_size})"
    )

    X = X[:valid_len, :]
    predictions = predictions[:valid_len]

    # 2) Perform Morris Analysis
    start_time_analysis = time.time()
    Si = morris_analyze.analyze(
        problem, X, predictions, print_to_console=False, num_levels=4
    )
    end_time_analysis = time.time()

    # 3) Extract Results into a clean DataFrame
    df = pd.DataFrame(
        {
            "parameter": problem["names"],
            "mu": Si["mu"],
            "mu_star": Si["mu_star"],
            "sigma": Si["sigma"],
            "mu_star_conf": Si["mu_star_conf"],
        }
    )

    print(
        f"Morris analysis done in {end_time_analysis - start_time_analysis:.2f} seconds."
    )
    return df


# --- Data Loading (Your Preferred Layout) ---

# Path & target
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="Data_after_KFold_LSSVR(ANOVA)").dropna()

# Automatically select the last column as target
target_column = df.columns[-1]

# Separate features
X = df.drop(columns=[target_column])

# Read the predictions text file into a flat numpy array
# header=None prevents pandas from accidentally eating your first row as a title!
y_df = pd.read_csv(r"C:\Users\Sam\Desktop\ML\data\predictions.txt", header=None)
y = y_df.iloc[:, 0].values  # Convert to a flat 1D array

print("Features Shape before alignment:", X.shape)
print("Predictions Shape before alignment:", y.shape)

# --- Execute and Copy to Clipboard ---
Morris_df = Morris_function(X=X, predictions=y)

print("\nFinal Results:")
print(Morris_df)

# Copy to clipboard for easy pasting into Excel
Morris_df.to_clipboard(index=False)
print("\nResults successfully copied to clipboard!")