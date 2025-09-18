import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from Metrics_regression import getAllMetric
import numpy as np

# --- Parameters ---
excel_path = r"D:\ML\Main_utils\Task\Concrete (elevated temperature)- Original result.xlsx"  # Replace with your Excel file path
sheet_name = "GBR_DATA"  # Replace with your sheet name
target_column = "CS"

# --- Load Data ---
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Features and Target ---
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# --- GBR Model ---
model = GradientBoostingRegressor(
    learning_rate=0.1, n_estimators=100, max_depth=3, alpha=0.9
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

df_train = pd.DataFrame({"y_train_real": y_train.values, "y_train_pred": y_pred_train})
df_test = pd.DataFrame({"y_test_real": y_test.values, "y_test_pred": y_pred_test})
df_all = pd.concat(
    [
        pd.DataFrame({"y_real": y_train.values, "y_pred": y_pred_train}),
        pd.DataFrame({"y_real": y_test.values, "y_pred": y_pred_test}),
    ],
    ignore_index=True,
)

print("\n📋 Performance Metrics Table:")
print(metrics_df)

# --- Load New Sheet for Additional Predictions ---
new_sheet_name = "Sheet2"  # Replace with your actual sheet name
df_new = pd.read_excel(excel_path, sheet_name=new_sheet_name)

X_new = df_new.drop(columns=[target_column])
y_new = df_new[target_column]
y_pred_new = model.predict(X_new)
df_new_pred = pd.DataFrame({"y_new_real": y_new.values, "y_new_pred": y_pred_new})

print("\n📋 Updated Performance Metrics Table (Including New Sheet):")
print(metrics_df)

# --- Define Actionable Variables (as per your task) ---
actionable_vars = [
    "Nano Silica",
    "Silica Fume",
    "Fly Ash",
    "Fine Aggregate",
    "Coarse Aggregate",
]

# --- CARE-inspired Counterfactual Analysis ---
low_strength_df = df[df[target_column] < 20].copy()
increase_rates = [0.10, 0.20, 0.35, 0.50]


def generate_counterfactuals(sample, model, increase_rate, actionable_vars):
    original_cs = sample[target_column]
    target_cs = original_cs * (1 + increase_rate)

    # Drop target column from sample
    x_cf = sample.drop(labels=[target_column]).copy()

    for _ in range(100):
        x_temp_df = pd.DataFrame([x_cf])
        pred_cs = model.predict(x_temp_df)[0]

        if pred_cs >= target_cs:
            break

        for var in actionable_vars:
            if var in x_cf:
                x_cf[var] += 0.01  # small step

    return x_cf


counterfactuals = []
for idx, row in low_strength_df.iterrows():
    for rate in increase_rates:
        cf = generate_counterfactuals(row, model, rate, actionable_vars)
        cf["Original_CS"] = row[target_column]
        cf["Target_CS"] = row[target_column] * (1 + rate)
        cf["Increase_Rate"] = rate
        counterfactuals.append(cf)


cf_df = pd.DataFrame(counterfactuals)
print("\n📊 Counterfactual Scenarios for Low Strength Samples:")
print(cf_df.head())


# --- CARE What-if Scenario Simulation ---
def simulate_care_scenarios(model, X, y_pred, increments=[0.10, 0.20, 0.35, 0.50]):
    scenario_results = []
    for inc in increments:
        target_strength = y_pred * (1 + inc)  # desired CS after % increase

        # Heuristic: scale actionable variables proportionally
        X_mod = X.copy()
        for var in actionable_vars:
            if var in X_mod.columns:
                X_mod[var] = X_mod[var] * (1 + inc * 0.5)

        # Predict with modified inputs
        y_pred_mod = model.predict(X_mod)

        # Calculate metrics vs target
        R, RMSE, MAE, RSE, SMAPE = getAllMetric(target_strength, y_pred_mod)

        scenario_results.append(
            {
                "Scenario": f"+{int(inc * 100)}%",
                "R2": R,
                "RMSE": RMSE,
                "MAE": MAE,
                "RSE": RSE,
                "SMAPE": SMAPE,
            }
        )

    return pd.DataFrame(scenario_results)


# --- Run CARE scenarios on test set ---
care_metrics_df = simulate_care_scenarios(model, X_new, y_pred_new)

# --- Merge with main metrics ---
metrics_df = pd.concat([metrics_df, care_metrics_df], ignore_index=True)

print("\n📋 Final Performance Metrics Table (With CARE What-if Scenarios):")
print(metrics_df)


# --- CARE What-if Scenario Predictions Table ---
def generate_care_predictions(model, X, increments=[0.10, 0.20, 0.35, 0.50]):
    prediction_table = pd.DataFrame(index=X.index)

    for inc in increments:
        X_mod = X.copy()
        for var in actionable_vars:
            if var in X_mod.columns:
                X_mod[var] = X_mod[var] * (1 + inc * 0.5)

        y_pred_mod = model.predict(X_mod)
        prediction_table[f"Predicted_CS_+{int(inc*100)}%"] = y_pred_mod

    return prediction_table


X_new_df = pd.DataFrame(X_new)
care_pred_table = generate_care_predictions(model, X_new_df)

# --- Combine with original predictions ---
care_pred_table["Original_Predicted_CS"] = y_pred_new
care_pred_table["Original_Actual_CS"] = y_new.values

# --- Display CARE prediction table ---
print("\n📊 CARE What-if Scenario Predictions:")
print(care_pred_table.head())
