# metrics_report_binary_autorun.py
"""
Binary classification metrics report generator.
This script loads predictions directly from a fixed file path,
computes metrics, and saves the report automatically.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)

# -------------------------
# Config: file paths
# -------------------------
# Input predictions file (first col = y_true, second col = y_pred)
file_path = Path(r"C:\Users\Sam\Desktop\ML\data\Data_err.npt")

# Output report file
out_path = Path(r"C:\Users\Sam\Desktop\ML\task\metrics_binary_report.xlsx")

# -------------------------
# Load predictions
# -------------------------
arr = np.loadtxt(file_path)
if arr.ndim == 1:
    arr = arr.reshape(1, -1)
if arr.shape[1] < 2:
    raise ValueError("Expected at least 2 columns: y_true, y_pred")

y_true = arr[:, 0].astype(int)
y_pred = arr[:, 1].astype(int)

# -------------------------
# Build metrics
# -------------------------
metrics = []
metrics.append(("N", len(y_true)))
metrics.append(("Accuracy", round(float(accuracy_score(y_true, y_pred)), 6)))
metrics.append(("Precision", round(float(precision_score(y_true, y_pred, zero_division=0)), 6)))
metrics.append(("Recall", round(float(recall_score(y_true, y_pred, zero_division=0)), 6)))
metrics.append(("F1", round(float(f1_score(y_true, y_pred, zero_division=0)), 6)))
try:
    mcc = matthews_corrcoef(y_true, y_pred)
    metrics.append(("MCC", round(float(mcc), 6)))
except Exception:
    metrics.append(("MCC", None))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
metrics.append(("ConfusionMatrix_shape", f"{cm.shape[0]}x{cm.shape[1]}"))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        metrics.append((f"CM[{i},{j}]", int(cm[i, j])))

# -------------------------
# Save report
# -------------------------
report_df = pd.DataFrame(metrics, columns=["Metric", "Value"])
report_df.insert(0, "No.", range(1, len(report_df) + 1))

try:
    report_df.to_excel(out_path, index=False)
    saved = out_path
except Exception:
    csvp = out_path.with_suffix(".csv")
    report_df.to_csv(csvp, index=False)
    saved = csvp

# Copy to clipboard (best effort)
try:
    report_df.to_clipboard(index=False)
except Exception:
    pass

# -------------------------
# Print preview
# -------------------------
print("✅ Binary classification report created.")
print("Saved to:", saved)
print("\nPreview:")
report_df.to_clipboard(index=False)
print(report_df.to_string(index=False))