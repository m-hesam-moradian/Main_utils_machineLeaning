"""
This module provides functions to calculate various statistical metrics for evaluating the performance of predictive models.
Metrics include RMSE, R-squared, MAE, MAPE, and others.
Author: ChatGPT
Date: 2024-06-10

Functions:
- getAllMetric(measured, predicted): Computes a comprehensive set of metrics comparing measured and predicted values.
"""

import numpy as np


def getAllMetric(measured, predicted):
    # Main Loop
    N = len(measured)
    S1, S2, S3, S4, S5, S6, S7, S8, S9 = 0, 0, np.zeros_like(measured), np.zeros_like(measured), 0, 0, 0, np.zeros_like(measured), 0
    R, R1, R2, R3, S10, S11, S12, S14, S15, S16,S17 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
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
        S2 += (predicted[i] - measured[i])
        S3[i] = (predicted[i] - measured[i])
        S12 += (measured[i] - predicted[i])
        S13 = (measured[i] - predicted[i])
        S10 += (2 * S13) / (predicted[i] + measured[i])
        S11 += S13 / measured[i]
        S14 += abs(S13) / abs(measured[i])
        S15 += abs(S13) / (abs((measured[i]) - np.mean(measured)))
        
        # IOA
        S16 =  S16+(measured[i]-predicted[i])**2
        S17 =  S17+((abs(predicted[i]-np.mean(measured)))+(abs(measured[i]-np.mean(measured))))**2

    for i in range(N - 1):
        # CP
        denominator = measured[i + 1] - measured[i]
        if denominator != 0:
            S1 += ((predicted[i + 1] - measured[i + 1])  2) / ((measured[i + 1] - measured[i])  2)
        else:
            # Handle the case where the denominator is zero (as per your use case)
            # You can choose to skip this iteration, set S1 to a default value, or handle it in another way
            pass
    ratio = measured / predicted
    num20 = np.logical_and(ratio < 1.2, ratio > 0.8)
    num10 = np.logical_and(ratio < 1.1, ratio > 0.9)
    n20 = np.sum(num20)
    n10 = np.sum(num10)

    MSE = M / N
    RMSE = np.sqrt(MSE)
    NRMSE = RMSE / N
    NMSE = MSE / N
    R = (R1 / np.sqrt(R2 * R3)) ** 2
    MAPE = S7 / N
    MDAPE = np.median(S8)
    VAF = (1 - (np.var(predicted - measured) / np.var(predicted))) * 100
    MAE = Z / N
    SI = RMSE / np.mean(measured)  # Scatter index
    RSR = RMSE / np.std(measured)  # ratio of RMSE to standard deviation
    CP = 1 - S1  # coefficient of persistence
    n20_index = n20 / N  # N20-INDEX
    n10_index = n10 / N  # N10-INDEX
    MBE = S2 / N  # Mean Error (Mean Bias Error)
    Tstate = np.sqrt((N - 1) * MBE ** 2 / (RMSE ** 2 - MBE ** 2))  # T statistic test
    U95 = 1.96 * np.sqrt(np.std(S3) ** 2 + RMSE ** 2)
    WAPE = Z / np.sum(np.abs(measured))  # Weighted Absolute Percentage Error (WAPE)
    SMAPE = (1 / N) * (Z / (np.sum(measured + predicted) / 2))  # Symmetric mean absolute percentage error
    FB = S10 / N  # Fractional Bias
    MNB = S11 / N  # Mean Normalized Bias
    MARE = S14 / N  # Mean Absolute Relative Error(Mean Magnitude Relative Error – MMRE)
    RAE = S15  # Relative Absolute Error
    MRAE = S15 / N  # Mean Relative Absolute Error
    PI = (1 / np.mean(measured)) * (RMSE / (np.sqrt(R) + 1))

    IOA = 1-(S16/S17)

    BIAS = np.mean(predicted)-np.mean(measured);

    NSE = 1-(M/R3)    
    # return RMSE,R, RSR, WAPE, MSE
    return RMSE,IOA

import math

def UI(list):
    num = math.sqrt(sum([(list[i] - list[i-1]) ** 2 for i in range(1, len(list))]) / (len(list) - 1))
    denum = math.sqrt(sum([x ** 2 for x in list[1:]]) / (len(list) - 1)) + math.sqrt(sum([x ** 2 for x in list[:-1]]) / (len(list) - 1))
    return num / denum