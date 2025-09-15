import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(file_path, sheet_name):
    """
    Load data from Excel and optionally separate target column.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()
    return df


# def normalize_minmax(X):
#     """
#     Min-Max normalization: scales features to [0,1].
#     """
#     scaler = MinMaxScaler()
#     X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
#     return X_norm, scaler


def normalize_standard(X):
    """
    Standard normalization: zero mean, unit variance.
    """
    scaler = StandardScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_norm, scaler


if __name__ == "__main__":
    # --- User settings ---
    file_path = (
        "D:\ML\Main_utils\Task\Original Dataset- Concrete (elevated temperature).xlsx"
    )
    sheet_name = "data"

    # --- Load data ---
    df = load_data(file_path, sheet_name)

    print("✅ Original data shape:", df.shape)

    # --- Normalize ---
    # X_minmax, minmax_scaler = normalize_minmax(df)
    # print("✅ Min-Max normalized data (first 5 rows):")
    # print(X_minmax.head())

    X_standard, standard_scaler = normalize_standard(df)
    print("✅ Standard normalized data (first 5 rows):")
    print(X_standard.head())
pd.DataFrame(X_standard)
# pd.DataFrame(X_minmax)
# pd.DataFrame(minmax_scaler)
# pd.DataFrame(standard_scaler)
