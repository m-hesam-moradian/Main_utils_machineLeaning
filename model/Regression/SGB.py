import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from Metrics_regression import getAllMetric

# --- Load data ---
sheet_name = "Data_after_KFold"
excel_path = r"D:\ML\Main_utils\task\Resource_utilization.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "cpu_utilization"

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

# --- Gradient Boosting Regressor Model ---
model = GradientBoostingRegressor(
    n_estimators=521,
    learning_rate=0.7073,
    max_depth=2,
    subsample=0.7521,
    min_samples_split=4,
    min_samples_leaf=3,
    max_features="sqrt",
    loss="squared_error",
    random_state=42,
)

# --- Save GBR parameters to DataFrame ---
gbr_params = {
    "n_estimators": model.n_estimators,
    "learning_rate": model.learning_rate,
    "max_depth": model.max_depth,
    "subsample": model.subsample,
    "min_samples_split": model.min_samples_split,
    "min_samples_leaf": model.min_samples_leaf,
    "max_features": model.max_features,
    "loss": model.loss,
}
horizantal_params_df = pd.DataFrame([gbr_params])
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
