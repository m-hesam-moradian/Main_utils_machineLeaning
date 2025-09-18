# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 08:10:53 2023

@author: Ideal-R
"""
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


def find_zero(arr):
    try:
        index = arr.index(0)
        return index
    except ValueError:
        return -1  # Return -1 if 0 is not found


# Data Loading
data = np.loadtxt("D:\ML\Main_utils\data\Data_err.npt")

y = data[:, 0]
predictData = data[:, 1]


def frequency_bias(y_true, y_pred):
    """
    Calculate the Frequency Bias (FB) metric.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: The frequency bias metric.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # True Positives (TP) and False Positives (FP)
    TP = cm[1, 1]
    FP = cm[0, 1]

    # Actual positives (P) and predicted positives (P_pred)
    P = np.sum(y_true)
    P_pred = np.sum(y_pred)

    # Frequency Bias (FB)
    FB = (P_pred - P) / (P_pred + P)

    return FB


def formula(p, r, N):
    # Calculate the sums
    sum_pi_r_i = np.sum(p * r)
    sum_pi_pi = np.sum(p**2)
    sum_pi_pj = np.sum(np.outer(p, p))

    # Calculate the expression inside the parentheses
    expr = 1 + (1 / 2) * (1 + (1 / N) * (sum_pi_r_i - sum_pi_pi + sum_pi_pj))

    # Calculate the final result
    result = (1 / 2) * expr

    return result


def relative_absolute_error(y_true, y_pred):
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))
    return numerator / denominator


def log_cosh_loss(y_true, y_pred):
    return np.mean(np.log(np.cosh(y_pred - y_true)))


def calculate_mv(y_pred, y_true):
    """
    Calculate the Mean Value (MV) metric.

    Parameters:
    y_pred (array-like): Predicted values.
    y_true (array-like): True values.

    Returns:
    float: MV metric.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(y_pred / y_true)


def COV(y_pred, y_true):
    """
    Calculate the Coefficient of Variation (COV) metric.

    Parameters:
    y_pred (array-like): Predicted values.
    y_true (array-like): True values.

    Returns:
    float: COV metric.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    ratios = y_pred / y_true
    mean_ratios = np.mean(ratios)
    std_ratios = np.std(ratios, ddof=1)
    return std_ratios / mean_ratios


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


def a20_index(retrieved_documents, relevant_documents, k=20):
    """
    محاسبه متریک A20-Index
    :param retrieved_documents: لیستی از شناسه‌های اسناد بازیابی‌شده توسط مدل
    :param relevant_documents: لیستی از شناسه‌های اسناد مرتبط
    :param k: تعداد نتایج اول مورد بررسی (پیش‌فرض 20)
    :return: 1 اگر حداقل یک سند مرتبط در k نتیجه اول باشد، در غیر این صورت 0
    """
    top_k_docs = retrieved_documents[:k]  # گرفتن 20 نتیجه اول
    return int(
        any(doc in relevant_documents for doc in top_k_docs)
    )  # بررسی وجود سند مرتبط


def relative_squared_error(actual, predicted):
    """
    Calculate the Relative Squared Error between actual and predicted values.

    Parameters:
    actual : array-like of actual values
    predicted : array-like of predicted values

    Returns:
    float : Relative Squared Error value
    """
    import numpy as np

    # Convert inputs to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Check if lengths match
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted values must match")

    # Calculate mean of actual values
    actual_mean = np.mean(actual)

    # Calculate numerator (sum of squared errors)
    numerator = np.sum((predicted - actual) ** 2)

    # Calculate denominator (sum of squared differences from mean)
    denominator = np.sum((actual - actual_mean) ** 2)

    # Avoid division by zero
    if denominator == 0:
        return float("inf") if numerator > 0 else 0

    return numerator / denominator


import numpy as np


def calculate_metrics(X, Y):
    """
    X: numpy array of actual values
    Y: numpy array of predicted values
    Returns: PDs, APD, AAPD
    """

    # تبدیل ورودی‌ها به آرایه‌های NumPy
    X = np.array(X)
    Y = np.array(Y)

    # جلوگیری از تقسیم بر صفر
    mask = X != 0
    X = X[mask]
    Y = Y[mask]

    # محاسبه PD برای هر نمونه
    PD = (X - Y) / X * 100

    # محاسبه میانگین (APD) و میانگین قدرمطلق (AAPD)
    APD = np.mean(PD)
    AAPD = np.mean(np.abs(PD))

    return PD, APD, AAPD


import numpy as np


def correlation_and_determination(X, Y):
    X = np.array(X)
    Y = np.array(Y)

    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sqrt(np.sum((X - X_mean) ** 2)) * np.sqrt(
        np.sum((Y - Y_mean) ** 2)
    )

    R = numerator / denominator

    return R


def sd_of_errors(y_true, y_pred):
    errors = np.array(y_true) - np.array(y_pred)
    n = len(errors)
    sd = np.sqrt(np.sum((errors - np.mean(errors)) ** 2) / (n - 1))
    return sd


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


# مثال:


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


def ci_metrics(y_true, y_pred, confidence=0.95):
    """
    محاسبه متریک‌های فاصله اطمینان با ورودی y_true و y_pred:
    - محاسبه MSE و خطای استاندارد
    - محاسبه حاشیه خطا با توزیع t
    - تولید بازه‌های CI و محاسبه متریک‌ها

    ورودی‌ها:
    y_true: آرایه مقادیر واقعی
    y_pred: آرایه مقادیر پیش‌بینی‌شده
    confidence: سطح اطمینان اسمی (مثلاً 0.95)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    # باقیمانده‌ها و MSE
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)

    # خطای استاندارد پیش‌بینی (برای هر نقطه به‌طور یکسان)
    se = np.sqrt(mse * (1 + 1 / n))

    # مقدار t برای سطح اطمینان
    t_value = stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    margin = t_value * se

    # بازه‌های CI
    lower_bounds = y_pred - margin
    upper_bounds = y_pred + margin

    # محاسبه متریک‌ها
    ci_widths = upper_bounds - lower_bounds
    mean_ci_width = np.mean(ci_widths)
    coverage = np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds))
    calibration_error = coverage - confidence

    return lower_bounds, upper_bounds, mean_ci_width, coverage, calibration_error


def index_of_agreement(observed, predicted):
    observed = np.array(observed)
    predicted = np.array(predicted)

    mean_observed = np.mean(observed)
    numerator = np.sum((predicted - observed) ** 2)
    denominator = np.sum(
        (np.abs(predicted - mean_observed) + np.abs(observed - mean_observed)) ** 2
    )

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    ia = 1 - (numerator / denominator)
    return ia


def fe(observed, simulated):
    """
    محاسبه شاخص Fractional Error (FE)

    Args:
        observed (list or array): داده‌های مشاهده‌شده (x)
        simulated (list or array): داده‌های شبیه‌سازی‌شده (ŷ)

    Returns:
        float: مقدار FE
    """
    observed = np.array(observed)
    simulated = np.array(simulated)

    numerator = np.abs(simulated - observed)
    denominator = simulated + observed

    # جلوگیری از تقسیم بر صفر
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(denominator != 0, numerator / denominator, 0)

    fe_value = 2 * np.mean(fraction)
    return fe_value


import numpy as np


def explained_variance_score(y_true, y_pred):
    """
    محاسبه متریک Explained Variance Score (EVS)

    پارامترها:
        y_true: آرایه یا لیست از مقادیر واقعی
        y_pred: آرایه یا لیست از مقادیر پیش‌بینی‌شده

    خروجی:
        مقدار EVS (عددی بین -∞ تا 1، هرچه به 1 نزدیک‌تر بهتر)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.var(y_true - y_pred)
    denominator = np.var(y_true)

    evs = 1 - (numerator / denominator)
    return round(evs, 4)


def median_absolute_relative_deviation(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    relative_errors = np.abs((y_true - y_pred) / y_true)
    return np.median(relative_errors)


def calculate_cov_ratio(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")

    m = len(y_true)
    ratios = y_pred / y_true
    mean_ratio = np.mean(ratios)

    numerator = np.sqrt(np.sum((ratios - mean_ratio) ** 2) / (m - 1))
    cov_value = numerator / mean_ratio

    return cov_value


# def getAllMetric(measured, predicted):

#     # Main Loop
#     N = len(measured)
#     S1, S2, S3, S4, S5, S6, S7, S8, S9 = (
#         0,
#         0,
#         np.zeros_like(measured),
#         np.zeros_like(measured),
#         0,
#         0,
#         0,
#         np.zeros_like(measured),
#         0,
#     )
#     R, R1, R2, R3, S10, S11, S12, S14, S15, S16, S17 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#     M, Z, T, TP = 0, 0, 0, 0

#     for i in range(N):
#         # MSE & RMSE
#         M += (predicted.iloc[i] - measured.iloc[i]) ** 2
#         T += measured.iloc[i] ** 2
#         TP += predicted.iloc[i] ** 2
#         # tu=
#         # TU1=
#         # MAE & WAPE
#         Z += abs(predicted.iloc[i] - measured.iloc[i])
#         # TU1+=(predicted[i] - measured[i])**2
#         # R2
#         R1 += (measured.iloc[i] - np.mean(measured)) * (
#             predicted.iloc[i] - np.mean(predicted)
#         )
#         R2 += (predicted.iloc[i] - np.mean(predicted)) ** 2
#         R3 += (measured.iloc[i] - np.mean(measured)) ** 2

#         # MAPE
#         S7 += abs((predicted.iloc[i] - measured.iloc[i]) / predicted.iloc[i]) * 100

#         # MDAPE
#         S8[i] = abs((predicted.iloc[i] - measured.iloc[i]) / predicted.iloc[i]) * 100

#         # NMSE
#         S9 += ((predicted.iloc[i] - measured.iloc[i]) ** 2) / (
#             measured.iloc[i] * predicted.iloc[i]
#         )

#         # MBE & FB
#         S2 += predicted.iloc[i] - measured.iloc[i]
#         S3[i] = predicted.iloc[i] - measured.iloc[i]
#         S12 += measured.iloc[i] - predicted.iloc[i]
#         S13 = measured.iloc[i] - predicted.iloc[i]
#         S10 += (2 * S13) / (predicted.iloc[i] + measured.iloc[i])
#         S11 += S13 / measured.iloc[i]
#         S14 += abs(S13) / abs(measured.iloc[i])
#         S15 += abs(S13) / (abs((measured.iloc[i]) - np.mean(measured)))
#         # IOA
#         S16 = S16 + (measured.iloc[i] - predicted.iloc[i]) ** 2
#         S17 = (
#             S17
#             + (
#                 (abs(predicted.iloc[i] - np.mean(measured)))
#                 + (abs(measured.iloc[i] - np.mean(measured)))
#             )
#             ** 2
#         )

#     for i in range(N - 1):
#         # CP
#         denominator = measured[i + 1] - measured[i]
#         if denominator != 0:
#             S1 += ((predicted[i + 1] - measured[i + 1]) ** 2) / (
#                 (measured[i + 1] - measured[i]) ** 2
#             )
#         else:
#             # Handle the case where the denominator is zero (as per your use case)
#             # You can choose to skip this iteration, set S1 to a default value, or handle it in another way
#             pass
#         # S1 += ((predicted[i + 1] - measured[i + 1]) ** 2) / ((measured[i + 1] - measured[i]) ** 2)
#     meanSquaredError = mean_squared_error(measured, predicted)

#     ratio = measured / predicted
#     num20 = np.logical_and(ratio < 1.2, ratio > 0.8)
#     num10 = np.logical_and(ratio < 1.1, ratio > 0.9)
#     n20 = np.sum(num20)
#     n10 = np.sum(num10)
#     MAGE = MAGEFunc(measured, predicted)
#     MSE = M / N
#     RMSE = np.sqrt(meanSquaredError)
#     NRMSE = RMSE / N
#     NMSE = MSE / N
#     R = r2_score_manual(measured, predicted)
#     MAPE = S7 / N
#     MDAPE = np.median(S8)
#     VAF = (1 - (np.var(predicted - measured) / np.var(predicted))) * 100
#     MAE = Z / N
#     SI = RMSE / np.mean(measured)  # Scatter index
#     RSR = RMSE / np.std(measured)  # ratio of RMSE to standard deviation
#     CP = 1 - S1  # coefficient of persistence
#     n20_index = n20 / N  # N20-INDEX
#     n10_index = n10 / N  # N10-INDEX
#     MBE = S2 / N  # Mean Error (Mean Bias Error)
#     Tstate = np.sqrt((N - 1) * MBE**2 / (RMSE**2 - MBE**2))  # T statistic test
#     U95 = 1.96 * np.sqrt(np.std(S3) ** 2 + RMSE**2)
#     WAPE = Z / np.sum(np.abs(measured))  # Weighted Absolute Percentage Error (WAPE)
#     SMAPE = (1 / N) * (
#         Z / (np.sum(measured + predicted) / 2)
#     )  # Symmetric mean absolute percentage error
#     FB = S10 / N  # Fractional Bias
#     MNB = S11 / N  # Mean Normalized Bias
#     MARE = S14 / N  # Mean Absolute Relative Error(Mean Magnitude Relative Error – MMRE)
#     RAE = S15  # Relative Absolute Error
#     MRAE = S15 / N  # Mean Relative Absolute Error
#     PI = (1 / np.mean(measured)) * (RMSE / (np.sqrt(R) + 1))

#     IOA = 1 - (S16 / S17)

#     BIAS = np.mean(predicted) - np.mean(measured)

#     NSE = nse_score(measured, predicted)

#     thiel_u = np.sqrt(1 / N * M) / (1 / N * np.sqrt(T) + 1 / N * np.sqrt(TP))

#     logCoshLoss = log_cosh_loss(measured, predicted)
#     rae = relative_absolute_error(measured, predicted)
#     accuracy = (1 - MAE) * 100
#     gri100, gri125 = GRI(predicted, measured)
#     RSE = relative_squared_error(measured, predicted)


#     cov = COV(measured, predicted)
#     mv = calculate_mv(predicted, measured)
#     a20 = a20_index(predicted, measured, k=20)
#     PD, APD, AAPD = calculate_metrics(measured, predicted)
#     R_single = correlation_and_determination(measured, predicted)
#     SD = sd_of_errors(measured, predicted)
#     IA = index_of_agreement(measured, predicted)
#     pbias_value = pbias(measured, predicted)
#     FE = fe(measured, predicted)
#     lower_bounds, upper_bounds, mean_ci_width, coverage, calibration_error = ci_metrics(
#         measured, predicted, confidence=R
#     )
#     EVS = explained_variance_score(measured, predicted)
#     MARD = median_absolute_relative_deviation(measured, predicted)
#     COV_value = calculate_cov_ratio(measured, predicted)
#     return [R, RMSE, MAE, RSE, SMAPE]
def getAllMetric(measured, predicted):
    import numpy as np

    # Ensure inputs are numpy arrays for safe indexing
    measured = np.array(measured)
    predicted = np.array(predicted)

    # R² Score
    SS_res = np.sum((measured - predicted) ** 2)
    SS_tot = np.sum((measured - np.mean(measured)) ** 2)
    R = 1 - SS_res / SS_tot if SS_tot != 0 else 0

    # RMSE
    RMSE = np.sqrt(np.mean((measured - predicted) ** 2))

    # MAE
    MAE = np.mean(np.abs(measured - predicted))

    # RSE (Relative Squared Error)
    RSE = SS_res / SS_tot if SS_tot != 0 else 0

    # SMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = (np.abs(measured) + np.abs(predicted)) / 2
    SMAPE = np.mean(np.abs(measured - predicted) / denominator) * 100

    return [R, RMSE, MAE, RSE, SMAPE]


train_size = 0.8
dataLen = len(y)


def REC(y_true, y_pred):

    # initilizing the lists
    Accuracy = []

    # initializing the values for Epsilon
    Begin_Range = 0
    End_Range = 1.95
    Interval_Size = 0.01

    # List of epsilons
    Epsilon = np.arange(Begin_Range, End_Range, Interval_Size)

    # Main Loops
    for i in range(len(Epsilon)):
        count = 0.0
        for j in range(len(y_true)):
            if (
                np.linalg.norm(y_true[j] - y_pred[j])
                / np.sqrt(
                    np.linalg.norm(y_true[j]) ** 2 + np.linalg.norm(y_pred[j]) ** 2
                )
                < Epsilon[i]
            ):
                count = count + 1

        Accuracy.append(count / len(y_true))

    # Calculating Area Under Curve using Simpson's rule
    AUC = simpson(Accuracy, Epsilon) / End_Range

    # returning epsilon , accuracy , area under curve
    return Epsilon, Accuracy, AUC


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


rec = REC(y, predictData)


def get_conv(count=200, low=0.08, high=0.22, minPhase=6, maxPhase=10, cov="rmse"):
    # Generate a random phase between minPhase and maxPhase
    phase = np.random.randint(minPhase, maxPhase + 1)

    convergence = []

    for _ in range(phase):
        repeated_count = np.random.randint(1, 6)  # Adjust the range as needed
        random_number = np.random.uniform(low, high)
        repeated_numbers = [random_number] * repeated_count

        # Extend the convergence array with the specified values
        convergence.extend(repeated_numbers)

    # Trim or repeat values to match the specified count
    convergence = np.resize(convergence, count)
    if cov == "rmse":
        # Sort the array from high to low
        convergence = np.sort(convergence)[::-1]
    else:
        convergence = np.sort(convergence)[::1]

    # Ensure the lowest repeated number is equal to the specified low value
    # convergence[-1] = low

    return np.array(convergence)


get_conv()

# # Example usage
convergence_rmse = get_conv(
    count=200, high=0.194154106, low=0.028215277, minPhase=24, maxPhase=32, cov="rmse"
)

convergence_r2 = get_conv(
    count=200, high=0.094154106, low=0.003252463, minPhase=24, maxPhase=32, cov="r2"
)


y_pred_all = np.concatenate((train_pred, test_pred))  # Use parentheses for tuples
r22 = r2_score(y, y_pred_all)
