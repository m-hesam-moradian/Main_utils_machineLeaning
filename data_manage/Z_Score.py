import pandas as pd
import numpy as np


def remove_outliers_zscore(df, threshold=3):
    """
    Removes rows with outliers based on Z-score and reports Z-scores.
    Additionally, removes the mirrored outliers (negative Z-scores).

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The Z-score threshold to classify as an outlier.

    Returns:
    pd.DataFrame: The DataFrame without outliers.
    pd.DataFrame: DataFrame containing the Z-scores.
    """
    # Calculate Z-scores
    z_scores = (df - df.mean()) / df.std()

    z_scores = z_scores.dropna()
    # Report Z-scores
    print("Z-scores:")
    print(z_scores)

    # Filter rows where all Z-scores are within the threshold, including mirrored Z-scores
    filtered_df = df[
        (z_scores.abs() < threshold).all(axis=1) & (z_scores > -threshold).all(axis=1)
    ]

    return filtered_df, z_scores
