import pandas as pd
import numpy as np
import time
from SALib.analyze import fast

start_time = time.time()

def Fast_function(X, predictions):
    """
    Inputs:
      - X: DataFrame (N_samples × D_features)
      - predictions: 1D numpy array of model outputs (continuous preferred, but here 0/1 labels)
    Output:
      - DataFrame with S1, S1_conf, ST, ST_conf for each parameter
    """
    column_names = X.columns
    X = np.array(X)
    predictions = np.array(predictions).astype(float).flatten()

    D = X.shape[1]
    problem = {
        "num_vars": D,
        "names": list(column_names),
        "bounds": [[np.min(X[:, i]), np.max(X[:, i])] for i in range(D)],
    }

    # --- Ensure predictions length is valid ---
    N = len(predictions)
    print(f"Original predictions length: {N}")
    if N % D != 0:
        valid_len = (N // D) * D
        print(f"Trimming predictions to length {valid_len} (nearest multiple of {D})")
        predictions = predictions[:valid_len]
    else:
        print("Predictions length is already a multiple of number of features.")

    # --- FAST analysis ---
    start_time = time.time()
    Si = fast.analyze(problem, predictions, print_to_console=False)
    end_time = time.time()

    df = pd.DataFrame({
        "parameter": problem["names"],
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    })

    print(f"FAST analysis done in {end_time - start_time:.2f} seconds.")
    return df


# --- Path & target ---
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="Data_after_KFold_ADAC")
target_column = df.columns[-1]

# Separate features and target
X = df.drop(columns=[target_column])

# --- Read predictions (single column of 0/1) ---
y_pred = pd.read_csv(r"C:\Users\Sam\Desktop\ML\data\predictions.txt", header=None)
y_pred = y_pred.iloc[:,0].astype(float).values   # ensure numeric array

print(df.head())

# Run FAST
Fast_df = Fast_function(X=X, predictions=y_pred)
Fast_df.to_clipboard(index=False)