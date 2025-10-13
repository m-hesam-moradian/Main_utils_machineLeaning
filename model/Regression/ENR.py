import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load data ---
sheet_name = "Data_after_KFold_DTR"
excel_path = (
    r"D:\ML\Main_utils\task\EI No. 5, Action Power-DTR-LGBR-ADAR-CPO-PRO-Data.xlsx"
)
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Power"

y = df[target_column].astype(float)

# --- Preprocess Features ---
categorical_cols = df.select_dtypes(include=["object"]).columns.drop(
    target_column, errors="ignore"
)
X = pd.get_dummies(
    df.drop(columns=[target_column]), columns=categorical_cols, drop_first=True
)

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, shuffle=False, random_state=42
)

# --- Define Elastic Net Model ---
model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

# --- Save ENR parameters to DataFrame ---
enr_params = {
    "alpha": model.alpha,
    "l1_ratio": model.l1_ratio,
    "random_state": model.random_state,
}
horizantal_params_df = pd.DataFrame([enr_params])
Vertical_params_df = pd.DataFrame(
    {
        "parameters": list(horizantal_params_df.columns),
        "values": list(horizantal_params_df.iloc[0]),
    }
)

# --- Fit Model ---
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X_scaled)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Split Test Predictions ---
mid_index = len(y_pred_test) // 2
y_test_first_half = y_test[:mid_index]
y_test_second_half = y_test[mid_index:]
y_pred_test_first_half = y_pred_test[:mid_index]
y_pred_test_second_half = y_pred_test[mid_index:]


# --- Build Metrics Table Using sklearn.metrics ---
def compute_metrics(y_true, y_pred):
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    R2 = r2_score(y_true, y_pred)
    return MAE, RMSE, R2


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
    MAE, RMSE, R2 = compute_metrics(y_true, y_pred)
    metrics_data["Set"].append(name)
    metrics_data["MAE"].append(MAE)
    metrics_data["RMSE"].append(RMSE)
    metrics_data["R2"].append(R2)

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
print("\n📊 Performance Metrics Table (ElasticNet Regression):")
print(metrics_df)
