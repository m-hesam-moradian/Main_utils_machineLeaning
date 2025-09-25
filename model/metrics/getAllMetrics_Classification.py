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
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    log_loss,
    accuracy_score,
)


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
    F2 = fbeta_score(measured, predicted, beta=2, average="weighted")
    # specificity = specificity_metric(measured, predicted)
    fm = calculate_metrics(measured, predicted)
    # logloss = log_loss(measured, predicted)
    # hss = heidke_skill_score(measured, predicted)
    # ci_brier, ci_ece = ci(measured, predicted, n_bins=10, plot=False)
    # g_mean = np.sqrt(specificity * recall)

    return [acc, precision_single, recall, f1, F2]
