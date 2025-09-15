import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


def rfe_selection(df, target_col="Vehicle Speed", n_features=5):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    model = LinearRegression()
    selector = RFE(model, n_features_to_select=n_features)
    selector.fit(X, y)

    selected_features = X.columns[selector.support_].tolist()
    return selected_features


if __name__ == "__main__":
    file_path = "Vehicle-Specific and Traffic _dataset.xlsx"
    df = pd.read_excel(file_path, sheet_name="sim_11").dropna()
    features = rfe_selection(df, "Vehicle Speed", n_features=5)
    print("✅ RFE Selected Features:", features)
