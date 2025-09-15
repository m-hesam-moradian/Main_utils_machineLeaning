import pandas as pd

# sys.path.insert(1, "E:/behnam/lib")
import numpy as np
import time
from SALib.analyze import fast
from sklearn.preprocessing import LabelEncoder

start_time = time.time()


def Fast_function(X, predictions):
    column_names = X.columns
    X = np.array(X)
    predictions = np.array(predictions)
    """ 
    ورودی‌ها: 
      - X: آرایه‌ی (N_samples × D_features) 
      - predictions: خروجی مدل به صورت یک آرایه‌ی 1D 
      - column_names: لیست اسامی ویژگی‌ها (پارامترها) 
    خروجی: 
      - DataFrame شامل S1, S1_conf, ST, ST_conf برای هر پارامتر 
    """

    D = X.shape[1]
    problem = {
        "num_vars": D,
        "names": list(column_names),
        "bounds": [[np.min(X[:, i]), np.max(X[:, i])] for i in range(D)],
    }

    # ۱) بررسی طول predictions
    N = len(predictions)
    print(f"Original predictions length: {N}")
    if N % D != 0:
        # کوتاه‌سازی تا نزدیک‌ترین مضرب
        valid_len = (N // D) * D
        print(f"Trimming predictions to length {valid_len} (nearest multiple of {D})")
        predictions = predictions[:valid_len]
    else:
        print("Predictions length is already a multiple of number of features.")

    # ۲) تحلیل FAST
    start_time = time.time()
    Si = fast.analyze(problem, predictions, print_to_console=False)
    end_time = time.time()

    # ۳) استخراج نتایج
    df = pd.DataFrame(
        {
            "parameter": problem["names"],
            "S1": Si["S1"],
            "S1_conf": Si["S1_conf"],
            "ST": Si["ST"],
            "ST_conf": Si["ST_conf"],
        }
    )

    print(f"FAST analysis done in {end_time - start_time:.2f} seconds.")
    return df


# Path & target
DATA_PATH = r"D:\ML\#M(XGBC&RFC)#O(LEOA)#RTIME#CI#WILCOXONN#FAST\#M(XGBC&RFC)#O(LEOA)#RTIME#CI#WILCOXONN#FAST\data\data.xlsx"
TARGET = "attack"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="data")

# Separate features and target
X = df.drop(columns=[TARGET])


# Read the text file into a DataFrame
y = pd.read_csv("D:\ML\Main_utils\predictions.txt")

# Display the DataFrame
print(df)


Fast_df = pd.DataFrame(Fast_function(X=X, predictions=y))
