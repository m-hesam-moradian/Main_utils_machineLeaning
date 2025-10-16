import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# Load data
data = np.loadtxt(r"D:\ML\ML\data\Data_err.npt")
y_real = data[:, 0]
y_pred = data[:, 1]

# Split into train/test
split_idx = int(len(y_real) * 0.8)
y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
y_pred_train, y_pred_test = y_pred[:split_idx], y_pred[split_idx:]


# Metric function
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# Compute metrics
metrics_all = get_metrics(y_real, y_pred)
metrics_train = get_metrics(y_real_train, y_pred_train)
metrics_test = get_metrics(y_real_test, y_pred_test)

# Create main metrics DataFrame
df_main = pd.DataFrame(
    [
        ["All", *metrics_all.values()],
        ["Train", *metrics_train.values()],
        ["Test", *metrics_test.values()],
    ],
    columns=["Set", "Accuracy", "Precision", "Recall", "F1", "MCC"],
)
# Compute per-class metrics using average=None
precision_per_class = precision_score(y_real, y_pred, average=None, zero_division=0)
recall_per_class = recall_score(y_real, y_pred, average=None, zero_division=0)
f1_per_class = f1_score(y_real, y_pred, average=None, zero_division=0)

# Accuracy per class: proportion of correct predictions for each class
accuracy_per_class = []
mcc_per_class = []
for cls in np.unique(y_real):
    idx = y_real == cls
    acc = accuracy_score(y_real[idx], y_pred[idx])
    mcc = matthews_corrcoef(y_real[idx], y_pred[idx])
    accuracy_per_class.append(acc)
    mcc_per_class.append(mcc)

# Build DataFrame
df_class = pd.DataFrame(
    {
        "Class": np.unique(y_real).astype(int),
        "Accuracy": accuracy_per_class,
        "Precision": precision_per_class,
        "Recall": recall_per_class,
        "F1": f1_per_class,
        # Remove MCC from here
    }
)


# Display both tables
print("Main Metrics Table:")
print(df_main.to_string(index=False))
print("\nPer-Class Metrics Table:")
print(df_class.to_string(index=False))
from sklearn.metrics import roc_curve, auc

# Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_real, y_pred)
roc_auc = auc(fpr, tpr)

# Create AUC column: empty strings except last row
auc_column = [""] * (len(fpr) - 1) + [round(roc_auc, 3)]

# Build DataFrame
roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "AUC": auc_column})

# Show the table
print(roc_df)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Confusion matrix
cm = confusion_matrix(y_real, y_pred)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)

# Print matrix
print("Confusion Matrix:")
print(cm_df)

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()
