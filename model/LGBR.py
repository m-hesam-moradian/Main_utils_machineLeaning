import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from getAllMetrics_Regression import getAllMetric

# --- Load data ---
sheet_name = "Data_after_KFold_SVC"
excel_path = r"D:\ML\Main_utils\task\EI_No_3__Optimal Scheduling_Classification_DTC_RFR_XGBC_HOA_DOA_Data.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Target"

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
    X, y, test_size=0.3, shuffle=False, random_state=42
)

# --- LGBMRegressor Model ---
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.7,
    min_child_samples=10,
    min_split_gain=0.01,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

# --- Save LGBM parameters to DataFrame ---
lgbm_params = {
    "n_estimators": model.n_estimators,
    "learning_rate": model.learning_rate,
    "max_depth": model.max_depth,
    "subsample": model.subsample,
    "min_child_samples": model.min_child_samples,
    "min_split_gain": model.min_split_gain,
    "colsample_bytree": model.colsample_bytree,
    "reg_alpha": model.reg_alpha,
    "reg_lambda": model.reg_lambda,
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
y_pred_test_first_half = y_pred_test
