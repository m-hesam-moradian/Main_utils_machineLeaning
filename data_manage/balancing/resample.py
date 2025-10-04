import pandas as pd
from sklearn.utils import resample


def balance_excel_data_oversample(
    input_path, label_column="label", sheet_name="Sheet1"
):
    # Step 1: Load Excel data
    df = pd.read_excel(input_path, sheet_name=sheet_name)

    # Step 2: Separate majority and minority classes
    df_majority = df[df[label_column] == 0]
    df_minority = df[df[label_column] == 1]

    # Step 3: Oversample minority class to match majority count
    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=len(df_majority), random_state=42
    )

    # Step 4: Combine and shuffle
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced


balanced_df = balance_excel_data_oversample(
    r"D:\ML\Main_utils\task\136_Seismic_ETC_RTHA, BO.xlsx",
    label_column="Class",
    sheet_name="Data after K-FOLD",
)
