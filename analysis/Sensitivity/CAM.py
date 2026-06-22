import numpy as np
import pandas as pd

def CAM_function(X, predictions):
    """
    Cosine Amplitude Method (CAM) sensitivity analysis.
    Inputs:
      - X: (N_samples × D_features) feature matrix
      - predictions: model outputs (N_samples,)
    Output:
      - DataFrame with CAM sensitivity indices per parameter
    """
    X = np.array(X)
    predictions = np.array(predictions).flatten()

    # --- Align lengths ---
    min_len = min(X.shape[0], len(predictions))
    X = X[:min_len, :]
    y = predictions[:min_len]

    D = X.shape[1]

    # Normalize outputs
    y_norm = (y - np.mean(y)) / np.std(y)

    # Sensitivity indices container
    cam_indices = []

    for j in range(D):
        # Normalize feature j
        xj = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])

        # Cosine transform (inner product)
        numerator = np.sum(y_norm * np.cos(np.pi * xj))
        denominator = np.sum(np.cos(np.pi * xj) ** 2)

        # CAM index for parameter j
        S_cam = (numerator ** 2) / (denominator * np.sum(y_norm ** 2))
        cam_indices.append(S_cam)

    df = pd.DataFrame({
        "parameter": [f"X{j+1}" for j in range(D)],
        "CAM_index": cam_indices
    })

    return df

# Example usage:
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\Data.xlsx", sheet_name="Data_after_KFold_LinearSVC")
target_column = df.columns[-1]
X = df.drop(columns=[target_column])

# Ensure predictions loads as a flat array
y = pd.read_csv(r"C:\Users\Sam\Desktop\ML\data\predictions.txt", header=None).squeeze()

cam_df = CAM_function(X, y)
cam_df.to_clipboard(index=False)