import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor  # Changed to AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

from Metrics_regression import getAllMetric

# --- Data Loading ---
sheet_name = "Data after K-Fold (ADAR)"
excel_path = r"D:\ML\Main_utils\Task\Global_AI_Content_Impact_Dataset.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Market Share of AI Companies (%)"

# --- Features and Target ---
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Handle NaN Values ---
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = y.fillna(y.mean())

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


# --- AdaBoostRegressor Model ---
model = AdaBoostRegressor(
    learning_rate=0.311,
    n_estimators=311,
    random_state=42,
    loss="linear",
)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Split Test Predictions ---
mid_index = len(y_pred_test) // 2
y_test_first_half = y_test.iloc[:mid_index]
y_test_second_half = y_test.iloc[mid_index:]
y_pred_test_first_half = y_pred_test[:mid_index]
y_pred_test_second_half = y_pred_test[mid_index:]


# --- Metrics Calculation ---
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse


# --- Build Metrics Table Using getAllMetric ---
metrics_data = {"Set": [], "R2": [], "RMSE": [], "MAE": [], "RSE": [], "SMAPE": []}
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test_first_half, y_pred_test_first_half),
    ("Test-Value", y_test_second_half, y_pred_test_second_half),
]

for name, y_true, y_pred in sets:
    R, RMSE, MAE, RSE, SMAPE = getAllMetric(y_true, y_pred)
    metrics_data["Set"].append(name)
    metrics_data["R2"].append(R)
    metrics_data["RMSE"].append(RMSE)
    metrics_data["MAE"].append(MAE)
    metrics_data["RSE"].append(RSE)
    metrics_data["SMAPE"].append(SMAPE)

metrics_df = pd.DataFrame(metrics_data)

# --- Additional DataFrames ---
df_train = pd.DataFrame({"y_train_real": y_train.values, "y_train_pred": y_pred_train})
df_test = pd.DataFrame({"y_test_real": y_test.values, "y_test_pred": y_pred_test})
df_all = pd.concat(
    [
        pd.DataFrame({"y_real": y_train.values, "y_pred": y_pred_train}),
        pd.DataFrame({"y_real": y_test.values, "y_pred": y_pred_test}),
    ],
    ignore_index=True,
)

# --- Output Results ---
print("\n📋 Performance Metrics Table:")
print(metrics_df)
print("\n📋 Training Data Predictions:")
print(df_train.head())
print("\n📋 Test Data Predictions:")
print(df_test.head())
print("\n📋 All Data Predictions:")
print(df_all.head())
