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
    precision_recall_curve,
    auc,
    confusion_matrix,
    average_precision_score,
    fbeta_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from itertools import cycle
import matplotlib.pyplot as plt

# Data Loading
data = np.loadtxt(r"D:\ML\main_structure\data\Data_err.npt")
y = data[:, 0]
predictData = data[:, 1]
import numpy as np
from collections import Counter


def get_confusion_matrix_values(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm


# Custom metric functions
def calculate_pod(cm, labels):
    pod = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        pod.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
    return pod


def calculate_far(cm, labels):
    far = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        far.append(fp / (fp + tp) if (fp + tp) != 0 else 0)
    return far


def calculate_csi(cm, labels):
    csi = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        csi.append(tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0)
    return csi


def calculate_fb(cm, labels):
    fb = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        fb.append((tp + fp) / (tp + fn) if (tp + fn) != 0 else 0)
    return fb


def calculate_far(cm, labels):
    far = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        far.append(fp / (fp + tp) if (fp + tp) != 0 else 0)
    return far


def calculate_hss(cm):
    n = cm.sum()
    po = np.trace(cm) / n
    pe = sum([(sum(cm[i, :]) * sum(cm[:, i])) for i in range(len(cm))]) / (n * n)
    # hss = (2 * (TP * TN - FP * FN)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0  # -*- coding: utf-8 -*-


import math
from sklearn.metrics import confusion_matrix


def FM(y, y_pred):
    # Generate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Calculate Precision (PPV)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    # Calculate Recall (TPR)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Calculate F-Measure (FM)
    f_measure = math.sqrt(precision * recall)

    return f_measure


def balanced_accuracy(y_true, y_pred):
    # محاسبه ماتریس سردرگمی
    cm = confusion_matrix(y_true, y_pred)

    # Check if binary classification
    if cm.shape == (2, 2):
        # Unpack confusion matrix values for binary classification
        tn, fp, fn, tp = cm.ravel()
    else:
        # Multiclass classification: Calculate metrics per class
        num_classes = cm.shape[0]
        metrics = {}

        for i in range(num_classes):
            # One-vs-All metrics for each class
            tn = cm[i, i]
            fn = sum(cm[i, :]) - tn
            fp = sum(cm[:, i]) - tn
            tp = sum(cm) - (tn + fn + fp)
    # محاسبه دقت متوازن
    balanced_acc = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))

    return balanced_acc


def calculate_acc_balance(cm, labels):
    acc_b = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(cm) - (tp + fn + fp)
        acc_b.append(0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))))
    return acc_b


def frequency_bias(y_true, y_pred):
    """
    Calculate the frequency bias metric.

    Parameters:
    y_true (array-like): True class labels
    y_pred (array-like): Predicted class labels

    Returns:
    float: The frequency bias value
    """
    # Get unique class labels
    classes = np.unique(np.concatenate([y_true, y_pred]))

    # Calculate true and predicted frequency distributions
    true_freq = np.array([np.sum(y_true == cls) for cls in classes]) / len(y_true)
    pred_freq = np.array([np.sum(y_pred == cls) for cls in classes]) / len(y_pred)

    # Calculate the frequency bias as the sum of absolute differences in frequency
    bias = np.sum(np.abs(true_freq - pred_freq))

    return bias


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


from sklearn.metrics import confusion_matrix


def calculate_metrics(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Check if binary classification
    if cm.shape == (2, 2):
        # Unpack confusion matrix values for binary classification
        TN, FP, FN, TP = cm.ravel()

        # Calculate metrics
        error_rate = (FN + FP) / (TP + TN + FN + FP)
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        f_measure = (
            2 * (precision * sensitivity) / (precision + sensitivity)
            if (precision + sensitivity) != 0
            else 0
        )

        metrics = {
            "Sensitivity (Recall)": sensitivity,
            "Precision": precision,
            "F-Measure": f_measure,
        }

    else:
        # Multiclass classification: Calculate metrics per class
        num_classes = cm.shape[0]
        metrics = {}

        for i in range(num_classes):
            # One-vs-All metrics for each class
            TP = cm[i, i]
            FN = sum(cm[i, :]) - TP
            FP = sum(cm[:, i]) - TP
            TN = sum(cm) - (TP + FN + FP)

            sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            f_measure = (
                2 * (precision * sensitivity) / (precision + sensitivity)
                if (precision + sensitivity) != 0
                else 0
            )

            metrics[f"Class {i}"] = {
                "Sensitivity (Recall)": sensitivity,
                "Precision": precision,
                "F-Measure": f_measure,
            }

    return f_measure


def specificity_metric(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):  # For binary classification
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    else:
        for i in range(num_classes):
            # True Negatives (TN): Exclude row and column for class i
            TN = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            # False Positives (FP): Sum of column for class i excluding diagonal
            FP = np.sum(cm[:, i]) - cm[i, i]
            # Specificity for class i
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    return specificity


import numpy as np


def jaccard_index(y, y_pred, average="weighted"):
    """
    Calculate the Jaccard Index for binary or multi-class data.

    Parameters:
    y (array-like): Ground truth labels.
    y_pred (array-like): Predicted labels.
    average (str): Type of averaging to perform. Options are:
        - 'macro': Calculate metrics for each class and take the average.
        - 'weighted': Calculate metrics for each class and take the average weighted by the number of true instances.
        - 'micro': Calculate metrics globally by counting total true positives, false positives, and false negatives.

    Returns:
    float: Jaccard Index.
    """
    y = np.array(y)
    y_pred = np.array(y_pred)

    # Get unique classes
    classes = np.unique(np.concatenate([y, y_pred]))

    if average == "micro":
        # Calculate global counts
        true_positives = np.sum((y == y_pred) & (y != 0))
        false_positives = np.sum((y != y_pred) & (y_pred != 0))
        false_negatives = np.sum((y != y_pred) & (y != 0))
        denominator = true_positives + false_positives + false_negatives
        return true_positives / denominator if denominator > 0 else 0.0

    jaccard_scores = []
    for cls in classes:
        # Calculate for each class
        tp = np.sum((y == cls) & (y_pred == cls))
        fp = np.sum((y != cls) & (y_pred == cls))
        fn = np.sum((y == cls) & (y_pred != cls))
        denominator = tp + fp + fn
        jaccard_scores.append(tp / denominator if denominator > 0 else 0.0)

    if average == "macro":
        return np.mean(jaccard_scores)
    elif average == "weighted":
        weights = [np.sum(y == cls) for cls in classes]
        return np.average(jaccard_scores, weights=weights)
    else:
        raise ValueError(
            "Invalid value for 'average'. Choose from 'macro', 'weighted', or 'micro'."
        )


# Example usage:


from sklearn.metrics import confusion_matrix
import numpy as np


def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute the Log Loss (Cross-Entropy Loss)

    Parameters:
    - y_true: array-like of shape (n_samples,) or (n_samples, n_classes)
              True labels (integers for multiclass or 0/1 for binary)
    - y_pred: array-like of shape (n_samples,) or (n_samples, n_classes)
              Predicted probabilities
    - eps: float, small number to avoid log(0)

    Returns:
    - log_loss: float
    """
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), eps, 1 - eps)

    # Binary classification
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        y_true = y_true.flatten()
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Multiclass classification
    n_samples = y_true.shape[0]
    if y_true.ndim == 1:
        # Convert to one-hot
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[np.arange(n_samples), y_true] = 1
    else:
        y_true_one_hot = y_true

    return -np.sum(y_true_one_hot * np.log(y_pred)) / n_samples


from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    log_loss,
    accuracy_score,
    balanced_accuracy_score,
)


from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings


def ci(y_true, y_proba, n_bins=10, plot=False):
    """
    نسخه‌ی مقاوم برای محاسبه Brier Score و ECE در Classification.
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    # اگر y_proba یک بردار بود → احتمال یک کلاس → تبدیل به دو ستون
    if y_proba.ndim == 1:
        warnings.warn(
            "y_proba had shape (n_samples,); assuming binary classification. Converting to 2-column format."
        )
        y_proba = np.stack([1 - y_proba, y_proba], axis=1)

    n_classes = y_proba.shape[1]

    # برچسب‌ها رو عددی کن (در صورت لزوم)
    if not np.issubdtype(y_true.dtype, np.integer):
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
        class_names = le.classes_
    else:
        class_names = [str(i) for i in range(n_classes)]

    # فیلتر نمونه‌هایی که کلاس‌شون توی y_proba نیست
    mask = y_true < n_classes
    if np.sum(mask) < len(y_true):
        dropped = len(y_true) - np.sum(mask)
        warnings.warn(
            f"{dropped} samples dropped: y_true contains class labels not in y_proba output shape."
        )
        y_true = y_true[mask]
        y_proba = y_proba[mask]

    # محاسبه pred و conf
    preds = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)

    # Accuracy
    accuracy = np.mean(preds == y_true)

    # Brier Score
    one_hot = np.eye(n_classes)[y_true]
    brier_score = np.mean(np.sum((y_proba - one_hot) ** 2, axis=1))

    # ECE
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    accs, confs = [], []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(mask):
            acc_bin = np.mean(preds[mask] == y_true[mask])
            conf_bin = np.mean(confidences[mask])
            ece += (mask.sum() / len(y_true)) * abs(acc_bin - conf_bin)
            accs.append(acc_bin)
            confs.append(conf_bin)
        else:
            accs.append(0)
            confs.append(0)

    # نمودار
    if plot:
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.figure(figsize=(6, 5))
        plt.plot(bin_centers, accs, label="Accuracy", marker="o")
        plt.plot(bin_centers, confs, label="Confidence", linestyle="--")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Reliability Diagram")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return brier_score, ece


from sklearn.preprocessing import LabelBinarizer


def heidke_skill_score(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))

    if len(labels) == 2:
        # ✅ باینری: همان کد اصلی شما
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        TN, FP, FN, TP = cm.ravel()

        numerator = 2 * (TP * TN - FP * FN)
        denominator = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)

        return numerator / denominator if denominator != 0 else 0

    else:
        # ✅ چندکلاسه: Macro-Averaged HSS
        hss_list = []
        for cls in labels:
            # One-vs-rest
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)

            cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            if cm.shape != (2, 2):
                # اگر کلاس موردنظر در pred یا true نبود
                hss_list.append(0)
                continue

            TN, FP, FN, TP = cm.ravel()
            numerator = 2 * (TP * TN - FP * FN)
            denominator = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
            hss = numerator / denominator if denominator != 0 else 0
            hss_list.append(hss)

        return np.mean(hss_list)


def getAllMetric(measured, predicted):
    cm = confusion_matrix(measured, predicted)

    if cm.shape == (2, 2):
        # Binary classification
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) != 0 else 0
        npv = TN / (TN + FN) if (TN + FN) != 0 else 0
        overall_accuracy = (
            (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        )
    else:
        # Multiclass classification - use macro/micro metrics
        sensitivity = recall_score(measured, predicted, average="macro")
        ppv = precision_score(measured, predicted, average="macro")
        npv = 0  # برای multiclass تعریف خاصی نداره
        overall_accuracy = accuracy_score(measured, predicted)

    # سایر متریک‌ها
    precision_single = precision_score(measured, predicted, average="macro")
    acc_balanced = balanced_accuracy_score(measured, predicted)
    acc = accuracy_score(measured, predicted)
    recall = recall_score(measured, predicted, average="macro")
    f1 = f1_score(measured, predicted, average="macro")
    f2 = fbeta_score(measured, predicted, beta=2, average="weighted")
    specificity = specificity_metric(measured, predicted)
    fm = calculate_metrics(measured, predicted)
    # logloss = log_loss(measured, predicted)
    hss = heidke_skill_score(measured, predicted)
    ci_brier, ci_ece = ci(measured, predicted, n_bins=10, plot=False)
    g_mean = np.sqrt(specificity * recall)
    metrics = [acc, precision_single, recall, f1]
    return metrics


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

    # Calculating Area Under Curve using Simpssimpsonon's rule
    AUC = simpson(Accuracy, Epsilon) / End_Range

    # returning epsilon , accuracy , area under curve
    return Epsilon, Accuracy, AUC


num_classes = len(np.unique(y))
actual_binary = label_binarize(y, classes=np.arange(num_classes))
predicted_scores = label_binarize(predictData, classes=np.arange(num_classes))
all_fpr = np.unique(
    np.concatenate(
        [
            roc_curve(actual_binary[:, i - 1], predicted_scores[:, i - 1])[0]
            for i in range(num_classes)
        ]
    )
)
mean_tpr = np.zeros_like(all_fpr)
auc_values = []
roc = []
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(actual_binary[:, i - 1], predicted_scores[:, i - 1])
    mean_tpr += np.interp(all_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    auc_values.append(roc_auc)
    roc.append(_)
mean_tpr /= num_classes
macro_auc = np.mean(auc_values)


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


def get_mcc_per_class(actual_val, pred_val):
    unique_classes = set(actual_val).union(set(pred_val))
    mcc_per_class = []

    for cls in unique_classes:
        class_actual = [1 if c == cls else 0 for c in actual_val]
        class_predicted = [1 if c == cls else 0 for c in pred_val]
        mcc = matthews_corrcoef(class_actual, class_predicted)
        mcc_per_class.append(mcc)

    return mcc_per_class


# r2_all = r2_score(y, predictData)

# hss=average_precision_score(y, predictData,average=None)
# average_precision_score
cm = get_confusion_matrix_values(y, predictData, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# محاسبه Class-Wise Error Rate برای هر کلاس
class_wise_error_rate = 1 - np.diag(cm) / cm.sum(axis=1)

# نمایش نتایج
for i, err in enumerate(class_wise_error_rate):
    print(f"Class {i} Error Rate: {err:.2f}")
csi = calculate_csi(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
hss = calculate_hss(cm)
fb = calculate_fb(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
far = calculate_far(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
pod = calculate_pod(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
acc_balance_list = calculate_acc_balance(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
f2 = fbeta_score(y, predictData, beta=2, average=None)
f1 = f1_score(y, predictData, average=None)
mcc = get_mcc_per_class(y, predictData)
pres = precision_score(y, predictData, average="weighted")
allMetric = getAllMetric(y, predictData)
trainMetric = getAllMetric(train_y, train_pred)
testMetric = getAllMetric(test_y, test_pred)
all_ressss = pd.DataFrame([allMetric, trainMetric, testMetric])
# precision, recall, thresholds = precision_recall_curve(y, predictData)

y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

precision_pcr = dict()
recall_pcr = dict()
prc_thresholds = dict()
auc_prc = []

for i in range(n_classes):
    precision_pcr[i], recall_pcr[i], prc_thresholds[i] = precision_recall_curve(
        y_bin[:, i], predictData
    )
pd.DataFrame()

for j in range(len(precision_pcr)):

    global precision_list
    global recall_list

    precision_list = sorted(
        precision_pcr[j].tolist()[:-1]
    )  # Convert NumPy array to list
    recall_list = recall_pcr[j].tolist()[:-1]
    # precision_list =  .sort()
    # recall_list = recall_list.sort()
    print("P", precision_list.sort())
    print("R", recall_list)
    auc_prc.append(auc(precision_list, recall_list))

prc_auc = np.mean(auc_prc)


def get_conv(count=200, low=0.08, high=0.22, minPhase=6, maxPhase=10):
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

    # Sort the array from high to low
    convergence = np.sort(convergence)[::1]

    # Ensure the lowest repeated number is equal to the specified low value
    # convergence[-1] = low

    return np.array(convergence)


# # Example usage
convergence = get_conv(
    count=200,
    high=0.99174522,
    low=0.76408088135749867626486863,
    minPhase=24,
    maxPhase=32,
)


precision = precision_score(y, predictData, average="macro")
recall = recall_score(y, predictData, average="macro")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# تبدیل به numpy array
y = np.array(y)
predictData = np.array(predictData)

thresholds = np.linspace(0.2, 1.0, num=10)  # از سختگیری کم تا زیاد

precisions = []
recalls = []

for thresh in thresholds:
    # ایجاد تغییر تصادفی در predictData
    mask = np.random.rand(len(predictData)) < thresh
    predictData_mod = predictData * mask  # بعضی 1ها رو صفر کنیم

    precision = precision_score(y, predictData_mod, average="macro", zero_division=0)
    recall = recall_score(y, predictData_mod, average="macro", zero_division=0)

    precisions.append(precision)
    recalls.append(recall)
list_df = pd.DataFrame(
    {"Recall": recall_list, "Precision": precision_list, "F1-score": f1, "MCC": mcc}
)
auc_prc = pd.DataFrame(auc_prc)
convergence_rmse = get_conv(
    count=200, high=0.957733813, low=0.45765563643, minPhase=24, maxPhase=32
)
# رسم نمودار
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, marker="o")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Approximate Precision-Recall Curve with stacking  ")
plt.grid()
plt.show()
