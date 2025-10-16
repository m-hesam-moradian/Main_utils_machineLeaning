import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


file_path = r"D:\ML\ML\task\BSE. No.13-Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="Encoded_Data")

target_column = "Cyberattack_Detected"
X = df.drop(columns=[target_column])


def calculate_vif(X, threshold=10.0, verbose=True):
    X = X.copy()
    while True:
        X_const = add_constant(X)
        vif = pd.DataFrame()
        vif["feature"] = X.columns
        vif["VIF"] = [
            variance_inflation_factor(X_const.values, i + 1) for i in range(X.shape[1])
        ]

        max_vif = vif["VIF"].max()
        if verbose:
            print(vif)
            print("=" * 40)

        if max_vif > threshold:
            drop_feature = vif.loc[vif["VIF"].idxmax(), "feature"]
            if verbose:
                print(f"📌 حذف ویژگی '{drop_feature}' با VIF = {max_vif:.2f}")
            X.drop(columns=[drop_feature], inplace=True)
        else:
            break

    return X, vif


selected_X, final_vif = calculate_vif(X, threshold=10.0)


print("✅ ویژگی‌های نهایی باقی‌مانده:")
print(selected_X.columns.tolist())
