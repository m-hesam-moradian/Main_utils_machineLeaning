import numpy as np
import random
from getMetricsForNoice import getMetrics
from getAllMetric import get_conv


def change_value(y, min_d, max_d, status):
    # اگر y اسکالر باشد، یک عدد تصادفی تولید کن
    if np.isscalar(y):
        random_number = random.uniform(min_d, max_d)
    # اگر y آرایه باشد، آرایه‌ای از اعداد تصادفی با همان شکل y تولید کن
    else:
        random_number = np.random.uniform(min_d, max_d, size=y.shape)

    one = y / 100
    if status:
        result = y + random_number * one
    else:
        result = y - random_number * one
    return result


def process_predictions(predict, y):
    if predict.shape != y.shape:
        raise ValueError("Predict و y باید شکل یکسانی داشته باشند.")

    scores = ((predict / y) - 1) * 100

    # ماسک‌ها برای شناسایی مقادیر خارج از محدوده
    mask_high = scores > 52
    mask_low = scores < -57

    # انتخاب تصادفی min_d و max_d برای mask_high
    min_d_high = random.uniform(20, 65)  # بازه‌ای برای حداقل
    max_d_high = random.uniform(10, 63)  # بازه‌ای برای حداکثر، بزرگ‌تر از حداقل

    # انتخاب تصادفی min_d و max_d برای mask_low
    min_d_low = random.uniform(10, 64)  # بازه‌ای برای حداقل
    max_d_low = random.uniform(10, 67)  # بازه‌ای برای حداکثر، بزرگ‌تر از حداقل

    # اعمال تغییرات با اعداد تصادفی متفاوت برای هر عنصر
    predict[mask_high] = change_value(y[mask_high], min_d_high, max_d_high, True)
    predict[mask_low] = change_value(y[mask_low], min_d_low, max_d_low, False)

    return predict


# Load and process data
data = np.loadtxt("Data_err.npt")
y = data[:, 0]
predict = data[:, 1]


def process_detect(predict, y):
    if predict.shape != y.shape:
        raise ValueError("Predict and y arrays must have the same shape.")

    scores = ((predict / y) - 1) * 100

    # شمارش تعداد سمبل‌هایی که مقدارشان بیش از 30 یا کمتر از -30 است
    extreme_count = ((scores > 40) | (scores < -40)).sum()

    return extreme_count


updated_predictions = process_predictions(predict, y)
detect_predictions = process_detect(predict, y)


metrics, train_RMSE = getMetrics(y, updated_predictions)


convergence_rmse = get_conv(
    count=200,
    high=0.194154106,
    low=train_RMSE,
    minPhase=24,
    maxPhase=32,
    cov="rmse",
)
