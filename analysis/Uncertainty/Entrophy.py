import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Load data
# -------------------------
data = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    sheet_name='Data_after_KFold_LinearSVC'
).head(1000)

# Target column
target_column = data.columns[-1]

# Features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# -------------------------
# Encode target if categorical
# -------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------
# Models
# -------------------------
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

svc = SVC(
    kernel='rbf',
    probability=True,
    random_state=42
)

# -------------------------
# Train models
# -------------------------
lr.fit(X, y_encoded)
svc.fit(X, y_encoded)

# -------------------------
# Predict probabilities
# -------------------------
proba_lr = lr.predict_proba(X)
proba_svc = svc.predict_proba(X)

# -------------------------
# Entropy function
# -------------------------
def entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)

# -------------------------
# Calculate uncertainty
# -------------------------
uncertainty_lr = entropy(proba_lr)
uncertainty_svc = entropy(proba_svc)

# -------------------------
# Add uncertainty columns
# -------------------------
data["Uncertainty_LR"] = uncertainty_lr
data["Uncertainty_SVC"] = uncertainty_svc

# -------------------------
# Display results
# -------------------------
print(data[["Uncertainty_LR", "Uncertainty_SVC"]].head())
data.to_clipboard()
# Optional: Save to Excel
# data.to_excel(
#     r"C:\Users\Sam\Desktop\ML\task\Uncertainty_LR_SVC.xlsx",
#     index=False
# )