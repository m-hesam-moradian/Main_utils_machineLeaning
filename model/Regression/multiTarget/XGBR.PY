import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# --- Load data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "UnderSampled"
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Define targets explicitly ---
target_columns = ["Electricity Price (USD/kWh)", " Electrical Power Demand (kW)"]
X = df.drop(columns=target_columns)

split_idx = int(len(df) * 0.8)

all_metrics = []
# --- Prepare wide-format predictions ---
predictions_wide = pd.DataFrame(index=df.index)

for target in target_columns:
    y = df[target]
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_all = model.predict(X)

    # Save metrics
    mid = len(y_test) // 2
    sets = [
        ("All", y, y_pred_all),
        ("Train", y_train, model.predict(X_train)),
        ("Test", y_test, model.predict(X_test)),
        ("Value", y_test[:mid], model.predict(X_test[:mid])),
        ("Test-Value", y_test[mid:], model.predict(X_test[mid:]))
    ]

    df_metrics = pd.DataFrame([{
        "Target": target,
        "Set": s,
        "MAE": mean_absolute_error(y_t, y_p),
        "RMSE": mean_squared_error(y_t, y_p) ** 0.5,
        "R2": r2_score(y_t, y_p)
    } for s, y_t, y_p in sets])
    all_metrics.append(df_metrics)

    # --- Add to wide-format predictions ---
    predictions_wide[f"y_real_{target}"] = y.values
    predictions_wide[f"y_pred_{target}"] = y_pred_all

# --- Combine metrics ---
final_metrics = pd.concat(all_metrics, ignore_index=True)
print(final_metrics)

# --- Show sample of wide-format predictions ---
print("\n📊 Sample of wide-format predictions:")
print(predictions_wide.head())

# --- Export to clipboard (wide format) ---
predictions_wide.to_clipboard(index=False, header=False)