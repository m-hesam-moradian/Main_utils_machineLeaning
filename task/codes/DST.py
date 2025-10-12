import numpy as np


def getAllMetric(measured, predicted):
    # Main Loop
    N = len(measured)
    S1, S2, S3, S4, S5, S6, S7, S8, S9 = (
        0,
        0,
        np.zeros_like(measured),
        np.zeros_like(measured),
        0,
        0,
        0,
        np.zeros_like(measured),
        0,
    )
    R, R1, R2, R3, S10, S11, S12, S14, S15, S16, S17 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    M, Z = 0, 0

    for i in range(N):
        # MSE & RMSE
        M += (predicted[i] - measured[i]) ** 2

        # MAE & WAPE
        Z += abs(predicted[i] - measured[i])

        # R2
        R1 += (measured[i] - np.mean(measured)) * (predicted[i] - np.mean(predicted))
        R2 += (predicted[i] - np.mean(predicted)) ** 2
        R3 += (measured[i] - np.mean(measured)) ** 2

        # MAPE
        S7 += abs((predicted[i] - measured[i]) / predicted[i]) * 100

        # MDAPE
        S8[i] = abs((predicted[i] - measured[i]) / predicted[i]) * 100

        # NMSE
        S9 += ((predicted[i] - measured[i]) ** 2) / (measured[i] * predicted[i])

        # MBE & FB
        S2 += predicted[i] - measured[i]
        S3[i] = predicted[i] - measured[i]
        S12 += measured[i] - predicted[i]
        S13 = measured[i] - predicted[i]
        S10 += (2 * S13) / (predicted[i] + measured[i])
        S11 += S13 / measured[i]
        S14 += abs(S13) / abs(measured[i])
        S15 += abs(S13) / (abs((measured[i]) - np.mean(measured)))

        # IOA
        S16 = S16 + (measured[i] - predicted[i]) ** 2
        S17 = (
            S17
            + (
                (abs(predicted[i] - np.mean(measured)))
                + (abs(measured[i] - np.mean(measured)))
            )
            ** 2
        )

    MSE = M / N
    RMSE = np.sqrt(MSE)

    IOA = 1 - (S16 / S17)

    return RMSE, IOA


import math


def UII(list):
    num = math.sqrt(sum([(list[i - 1] - list[i]) ** 2 for i in range(1, len(list))]))
    denum = math.sqrt(sum([x**2 for x in list]))
    return num / denum


def U_Oracle(list):
    num = math.sqrt(
        sum([((list[i - 1] - list[i]) / list[i - 1]) ** 2 for i in range(1, len(list))])
    )
    denum = math.sqrt(
        sum([((list[i] - list[i - 1]) / list[i - 1]) ** 2 for i in range(1, len(list))])
    )
    return num / denum


list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


print(UII(list))
print(U_Oracle(list))
ytrs = []
mesData = []
RMSE_values = []
IOA_values = []
array_of_floats_mes = [1.3, 6, 2, 2, 9]
modelTmp = {
    "XGB": "P3:P677",
    "CAT": "AD3:AD677",
    # "STOC": "C3:C918",
}


def dempster_of_models(models, measures):
    for model in models:

        preData = model["y_pred"]
        RMSE, IOA = getAllMetric(measures, preData)
        RMSE_values.append(RMSE)
        IOA_values.append(IOA)
        ytrs.append(preData)
    ytr1 = ytrs[0]
    sumIOA_Vals = sum(IOA_values)
    m_BI = [IOA / sumIOA_Vals for IOA in IOA_values]
    WTH = sum([1 / RMSE_val for RMSE_val in RMSE_values])
    m_CI = [(1 / RMSE) / WTH for RMSE in RMSE_values]
    weights = []
    for i in range(len(m_BI)):
        weights.append((m_BI[i] + m_CI[i]) / 2)
    N = len(ytr1)
    ytr_en = np.zeros(N)
    for i in range(N):
        ensemble_prediction = sum(weights[j] * ytrs[j][i] for j in range(len(weights)))
        ytr_en[i] = ensemble_prediction

    return ytr_en, weights


import pandas as pd

# Load Excel file (update the path to your real file)
file_path = r"D:\ML\Main_utils\task\EI No. 5, Action Power-DTR-LGBR-ADAR-CPO-PRO-Data.xlsx"  # <-- Replace with your actual file
sheet_name = "DTS-Data"

# Read the Excel sheet
df = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1])

models = []
df = df[~(df == 0.0).any(axis=1)]
# Group every two columns (y_real/y_pred pairs) under the same model
for model_name in df.columns.levels[0]:
    y_real = pd.to_numeric(df[model_name]["y_real"], errors="coerce").dropna().tolist()
    y_pred = pd.to_numeric(df[model_name]["y_pred"], errors="coerce").dropna().tolist()

    model_data = {
        "name": model_name,
        "y_real": y_real,
        "y_pred": y_pred,
    }
    models.append(model_data)


# MANUAL: Choose your model group
selected_group_name = "LGBR"

# Filter models for selected group
group_models = [model for model in models if selected_group_name in model["name"]]

# Use the y_real from the first model as the ground truth
measures = group_models[0]["y_real"]
y_real = pd.DataFrame(measures)
# Run ensemble fusion
ensemble_prediction, model_weights = dempster_of_models(group_models, measures)

# Build the weights table
weights_table = pd.DataFrame(
    {"Model Name": [model["name"] for model in group_models], "Weight": model_weights}
)

# Store in variables for later use
ensemble_result = {
    "name": selected_group_name,
    "prediction": ensemble_prediction,
    "weights_table": weights_table,
}

# Optional: print or access
print(f"--- {ensemble_result['name']} Ensemble Summary ---")
print("Sample predictions:", ensemble_result["prediction"][:5])
print(ensemble_result["weights_table"])
