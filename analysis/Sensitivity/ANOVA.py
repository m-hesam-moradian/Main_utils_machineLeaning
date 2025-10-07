from scipy.stats import f_oneway
import numpy as np
import pandas as pd

DATA_PATH = r"D:\ML\Main_utils\task\EI_No_3__Optimal Scheduling_Classification_DTC_RFR_XGBC_HOA_DOA_Data.xlsx"
TARGET = "Target"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="DATA_Normalized")

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]


def anova_features_dataset(x, y):
    all = []
    x = np.array(x)
    print("XXX", x)
    for i in range(x.shape[1]):
        f, p = f_oneway(x[:, i - 1], y)
        all.append([f, p])
    all = np.array(all)
    return all


ANOVA = anova_features_dataset(X, y)
