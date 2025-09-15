import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.integrate import simpson
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    precision_score,
)
from scipy import stats


def MAGEFunc(predictions, actuals):
    # Ensure the inputs are numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate the absolute errors
    absolute_errors = np.abs(predictions - actuals)

    # Compute the mean absolute gross error (MAGE)
    mage = np.mean(absolute_errors)

    return mage


def r2_score_manual(y_true, y_pred):
    """
    محاسبه ضریب تعیین (R^2)

    :param y_true: لیست یا آرایه‌ای از مقادیر واقعی
    :param y_pred: لیست یا آرایه‌ای از مقادیر پیش‌بینی‌شده
    :return: مقدار R^2
    """
    # تبدیل لیست‌ها به آرایه‌های numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # محاسبه میانگین مقادیر واقعی
    mean_true = np.mean(y_true)

    # محاسبه مجموع مربعات خطا (SSE) و مجموع مربعات کل (SST)
    ss_res = np.sum((y_true - y_pred) ** 2)  # مجموع مربعات خطا
    ss_tot = np.sum((y_true - mean_true) ** 2)  # مجموع مربعات کل

    # محاسبه R^2
    r2 = 1 - (ss_res / ss_tot)
    return r2


def nse_score(Q_obs, Q_sim):
    """
    محاسبه متریک Nash-Sutcliffe Efficiency (NSE)

    :param Q_obs: لیست یا آرایه‌ای از مقادیر مشاهده‌شده
    :param Q_sim: لیست یا آرایه‌ای از مقادیر شبیه‌سازی‌شده
    :return: مقدار NSE
    """
    # محاسبه میانگین مقادیر مشاهده‌شده
    mean_obs = np.mean(Q_obs)

    # محاسبه بروز خطاها
    numerator = np.sum((Q_obs - Q_sim) ** 2)
    denominator = np.sum((Q_obs - mean_obs) ** 2)

    # محاسبه و بازگشت NSE
    NSE = 1 - (numerator / denominator)
    return NSE


def GRI(y_true, y_pred):
    """
    محاسبه GR100 و GR125 برای مقایسه پیش‌بینی با مقدار واقعی

    پارامترها:
        y_true: لیست یا آرایه‌ی مقادیر واقعی
        y_pred: لیست یا آرایه‌ی مقادیر پیش‌بینی‌شده

    خروجی:
        دیکشنری شامل GR100 و GR125 به صورت درصد (مثلاً 85.0 به معنی 85%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ratio = y_pred / y_true

    gr100 = np.mean(ratio <= 1.00) * 100  # ≤ 100%
    gr125 = np.mean(ratio <= 1.25) * 100  # ≤ 125%

    return round(gr100, 2), round(gr125, 2)


def pbias(observed, simulated):
    """
    محاسبه PBIAS یا IAE بین داده‌های مشاهده‌شده و شبیه‌سازی‌شده

    Args:
        observed (list or array): لیست یا آرایه‌ای از داده‌های مشاهده‌شده (x)
        simulated (list or array): لیست یا آرایه‌ای از داده‌های شبیه‌سازی‌شده (ŷ)

    Returns:
        float: مقدار PBIAS
    """
    observed = np.array(observed)
    simulated = np.array(simulated)

    numerator = np.sum(observed - simulated)
    denominator = np.sum(simulated)

    if denominator == 0:
        raise ZeroDivisionError(
            "مجموع داده‌های شبیه‌سازی‌شده صفر است، نمی‌توان تقسیم انجام داد."
        )

    return numerator / denominator


def relative_absolute_error(y_true, y_pred):
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))
    return numerator / denominator


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
    M, Z, T, TP = 0, 0, 0, 0

    for i in range(N):
        # MSE & RMSE
        M += (predicted[i] - measured[i]) ** 2
        T += measured[i] ** 2
        TP += predicted[i] ** 2
        # tu=
        # TU1=
        # MAE & WAPE
        Z += abs(predicted[i] - measured[i])
        # TU1+=(predicted[i] - measured[i])**2
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

    for i in range(N - 1):
        # CP
        denominator = measured[i + 1] - measured[i]
        if denominator != 0:
            S1 += ((predicted[i + 1] - measured[i + 1]) ** 2) / (
                (measured[i + 1] - measured[i]) ** 2
            )
        else:
            # Handle the case where the denominator is zero (as per your use case)
            # You can choose to skip this iteration, set S1 to a default value, or handle it in another way
            pass
        # S1 += ((predicted[i + 1] - measured[i + 1]) ** 2) / ((measured[i + 1] - measured[i]) ** 2)
    meanSquaredError = mean_squared_error(measured, predicted)

    ratio = measured / predicted
    # num20 = np.logical_and(ratio < 1.2, ratio > 0.8)
    num10 = np.logical_and(ratio < 1.1, ratio > 0.9)
    # n20 = np.sum(num20)
    n10 = np.sum(num10)
    # MAGE = MAGEFunc(measured, predicted)
    # MSE = M / N
    RMSE = np.sqrt(meanSquaredError)
    # NRMSE = RMSE / N
    # NMSE = MSE / N
    # R = r2_score_manual(measured, predicted)
    # MAPE = S7 / N
    # MDAPE = np.median(S8)
    # VAF = (1 - (np.var(predicted - measured) / np.var(predicted))) * 100
    MAE = Z / N
    # SI = RMSE / np.mean(measured)  # Scatter index
    # RSR = RMSE / np.std(measured)  # ratio of RMSE to standard deviation
    # CP = 1 - S1  # coefficient of persistence
    # n20_index = n20 / N  # N20-INDEX
    # n10_index = n10 / N  # N10-INDEX
    # MBE = S2 / N  # Mean Error (Mean Bias Error)
    # # Tstate = np.sqrt((N - 1) * MBE**2 / (RMSE**2 - MBE**2))  # T statistic test
    # # U95 = 1.96 * np.sqrt(np.std(S3) ** 2 + RMSE**2)
    # # WAPE = Z / np.sum(np.abs(measured))  # Weighted Absolute Percentage Error (WAPE)
    # # SMAPE = (1 / N) * (
    # #     Z / (np.sum(measured + predicted) / 2)
    # # )  # Symmetric mean absolute percentage error
    # FB = S10 / N  # Fractional Bias
    # MNB = S11 / N  # Mean Normalized Bias
    # MARE = S14 / N  # Mean Absolute Relative Error(Mean Magnitude Relative Error – MMRE)
    # RAE = S15  # Relative Absolute Error
    # MRAE = S15 / N  # Mean Relative Absolute Error
    # PI = (1 / np.mean(measured)) * (RMSE / (np.sqrt(R) + 1))

    # IOA = 1 - (S16 / S17)

    # BIAS = np.mean(predicted) - np.mean(measured)

    # NSE = nse_score(measured, predicted)

    # thiel_u = np.sqrt(1 / N * M) / (1 / N * np.sqrt(T) + 1 / N * np.sqrt(TP))

    # logCoshLoss = log_cosh_loss(measured, predicted)
    rae = relative_absolute_error(measured, predicted)
    # accuracy = (1 - MAE) * 100
    # gri100, gri125 = GRI(predicted, measured)
    # rse = relative_squared_error(measured, predicted)

    # cov = COV(measured, predicted)
    # mv = calculate_mv(predicted, measured)
    # a20 = a20_index(predicted, measured, k=20)
    # PD, APD, AAPD = calculate_metrics(measured, predicted)
    # R_single = correlation_and_determination(measured, predicted)
    # SD = sd_of_errors(measured, predicted)
    # IA = index_of_agreement(measured, predicted)
    # pbias_value = pbias(measured, predicted)
    # FE = fe(measured, predicted)
    # lower_bounds, upper_bounds, mean_ci_width, coverage, calibration_error = ci_metrics(
    #     measured, predicted, confidence=R
    # )
    # EVS = explained_variance_score(measured, predicted)
    # MARD = median_absolute_relative_deviation(measured, predicted)
    # COV_value = calculate_cov_ratio(measured, predicted)
    return [R, RMSE, MAE, rae, n10]


def find_zero(arr):
    try:
        index = arr.index(0)
        return index
    except ValueError:
        return -1  # Return -1 if 0 is not found


# Data Loading
data = np.loadtxt("D:\ML\Main_utils\Data_err.npt")

y = data[:, 0]
predictData = data[:, 1]


train_size = 0.8
dataLen = len(y)


split_index_train_test = round(dataLen * train_size)

test_count = dataLen - split_index_train_test

split_index_test_val = round(test_count / 2)

train_y = y[:split_index_train_test]
train_pred = predictData[:split_index_train_test]


test_y = y[split_index_train_test:]
test_pred = predictData[split_index_train_test:]

test_val_y = test_y[:split_index_test_val]
test_val_pred = test_pred[:split_index_test_val]

test_test_y = test_y[split_index_test_val:]
test_test_pred = test_pred[split_index_test_val:]


# r2_all = r2_score(y, predictData)
index_of_zero_P = find_zero(list(predictData))
index_of_zero_y = find_zero(list(y))
allMetric = getAllMetric(y, predictData)
trainMetric = getAllMetric(train_y, train_pred)
testMetric = getAllMetric(test_y, test_pred)
testValMetric = getAllMetric(test_val_y, test_val_pred)
testTestMetric = getAllMetric(test_test_y, test_test_pred)

all_ressss = pd.DataFrame(
    [allMetric, trainMetric, testMetric, testValMetric, testTestMetric]
)
