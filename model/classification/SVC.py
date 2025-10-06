import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from getAllMetrics_Classification import getAllMetric

# --- Load data ---
sheet_name = "Data_after_KFold_SVC"
excel_path = r"D:\ML\Main_utils\task\EI_No_3__Optimal Scheduling_Classification_DTC_RFR_XGBC_HOA_DOA_Data.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Target"

# --- Encode Target Variable ---
if df[target_column].dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(df[target_column])
else:
    median_value = df[target_column].median()
    y = (df[target_column] > median_value).astype(int)

# --- Preprocess Features ---
categorical_cols = df.select_dtypes(include=["object"]).columns.drop(
    target_column, errors="ignore"
)
X = pd.get_dummies(
    df.drop(columns=[target_column]), columns=categorical_cols, drop_first=True
)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False, random_state=42
)

# --- SVC Model ---

model = SVC(
    C=0.1,
    degree=1,
    coef0=0,
    tol=1,
    max_iter=100,
    cache_size=50,
)
# --- Save SVC parameters to DataFrame ---
svc_params = {
    "C": model.C,
    "degree": model.degree,
    "coef0": model.coef0,
    "tol": model.tol,
    "max_iter": model.max_iter,
    "cache_size": model.cache_size,
}
horizantal_params_df = pd.DataFrame([svc_params])
Vertical_params_df = pd.DataFrame(
    {
        "parameters": list(horizantal_params_df.columns),
        "values": list(horizantal_params_df.iloc[0]),
    }
)


model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Split Test Predictions ---
mid_index = len(y_pred_test) // 2
y_test_first_half = y_test[:mid_index]
y_test_second_half = y_test[mid_index:]
y_pred_test_first_half = y_pred_test[:mid_index]
y_pred_test_second_half = y_pred_test[mid_index:]

# --- Build Metrics Table Using getAllMetric ---
metrics_data = {
    "Set": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "F2": [],
}
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test_first_half, y_pred_test_first_half),
    ("Test-Value", y_test_second_half, y_pred_test_second_half),
]

for name, y_true, y_pred in sets:
    Accuracy, Precision, Recall, F1, F2 = getAllMetric(y_true, y_pred)
    metrics_data["Set"].append(name)
    metrics_data["Accuracy"].append(Accuracy)
    metrics_data["Precision"].append(Precision)
    metrics_data["Recall"].append(Recall)
    metrics_data["F1"].append(F1)
    metrics_data["F2"].append(F2)

metrics_df = pd.DataFrame(metrics_data)

# --- Create DataFrames for real vs predicted ---
df_train = pd.DataFrame({"y_train_real": y_train, "y_train_pred": y_pred_train})
df_test = pd.DataFrame({"y_test_real": y_test, "y_test_pred": y_pred_test})
df_all = pd.concat(
    [
        pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train}),
        pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test}),
    ],
    ignore_index=True,
)

# --- Print Metrics Table ---
print("\n📋 Performance Metrics Table (SVC):")
print(metrics_df)
