import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from Metrics_regression import getAllMetric
from lightgbm import LGBMRegressor


# --- Load data ---
sheet_name = "Data_after_KFold_LGBR"
excel_path = (
    r"D:\ML\Main_utils\task\EI No. 5, Action Power-DTR-LGBR-ADAR-CPO-PRO-Data.xlsx"
)
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Power"


# --- Encode Target Variable if needed ---
if df[target_column].dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(df[target_column])
else:
    y = df[target_column].astype(float)

# --- Preprocess Features ---
categorical_cols = df.select_dtypes(include=["object"]).columns.drop(
    target_column, errors="ignore"
)
X = pd.get_dummies(
    df.drop(columns=[target_column]), columns=categorical_cols, drop_first=True
)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

model = LGBMRegressor(
    n_estimators=296,
    learning_rate=0.138,
    max_depth=6,
    subsample=0.371,
    min_child_samples=13,
    random_state=42,
)

# --- Save LGBM parameters to DataFrame ---
lgbm_params = {
    "n_estimators": model.n_estimators,
    "learning_rate": model.learning_rate,
    "max_depth": model.max_depth,
    "subsample": model.subsample,
    "min_child_samples": model.min_child_samples,
}
horizantal_params_df = pd.DataFrame([lgbm_params])
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
    "MAE": [],
    "RMSE": [],
    "R2": [],
}
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test_first_half, y_pred_test_first_half),
    ("Test-Value", y_test_second_half, y_pred_test_second_half),
]

for name, y_true, y_pred in sets:
    MAE, RMSE, R2 = getAllMetric(y_true, y_pred)
    metrics_data["R2"].append(R2)
    metrics_data["Set"].append(name)
    metrics_data["MAE"].append(MAE)
    metrics_data["RMSE"].append(RMSE)

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
print("\n📊 Performance Metrics Table (GradientBoostingRegressor):")
print(metrics_df)
