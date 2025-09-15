def get_X_y(df, target_col):
    """
    Splits the DataFrame into features (X) and target (y)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
