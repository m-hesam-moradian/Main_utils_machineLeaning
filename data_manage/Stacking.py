import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# =====================================================================
# 1. METRICS FUNCTION (Adapted for Regression)
# =====================================================================
def build_regression_reports(y_real, y_pred):
    
    def get_metrics(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        
        # Core regression metrics
        r2 = r2_score(y_true, y_hat)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        mae = mean_absolute_error(y_true, y_hat)
        
        # MAPE (Mean Absolute Percentage Error) - avoiding division by zero
        non_zero = y_true != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((y_true[non_zero] - y_hat[non_zero]) / y_true[non_zero])) * 100
        else:
            mape = np.nan
            
        return {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE (%)": mape
        }

    # --- Train/Test split
    split = int(len(y_real) * 0.8)
    y_real_train, y_real_test = y_real[:split], y_real[split:]
    y_pred_train, y_pred_test = y_pred[:split], y_pred[split:]

    cols = ["Set", "R2", "RMSE", "MAE", "MAPE (%)"]

    df_metrics = pd.DataFrame([
        ["All", *get_metrics(y_real, y_pred).values()],
        ["Train", *get_metrics(y_real_train, y_pred_train).values()],
        ["Test", *get_metrics(y_real_test, y_pred_test).values()],
    ], columns=cols)

    return df_metrics

# =====================================================================
# 2. DATA LOADING & REGRESSION STACKING FUSION
# =====================================================================

# Load data
data = np.loadtxt(r"C:\Users\Sam\Desktop\ML\data\Data_err.npt")
y_real = data[:, 0]
y_pred_lgbr = data[:, 1]
y_pred_sgb = data[:, 2]

# --- SANITY CHECK: Verify base model R2 scores ---
print("\n--- Base Model Sanity Check ---")
print(f"Base 1 (LGBR) R2: {r2_score(y_real, y_pred_lgbr):.4f}")
print(f"Base 2 (SGB) R2:  {r2_score(y_real, y_pred_sgb):.4f}")
print("-------------------------------\n")

# 1. Stack the raw continuous predictions
# No need for OneHotEncoder in Regression!
X_meta = np.column_stack((y_pred_lgbr, y_pred_sgb))

# 2. Initialize the Meta-Model for Regression
# LinearRegression is the standard, most robust choice for regression stacking
meta_model = LinearRegression()

print("Training Meta-Model...")

# 3. Direct fit and predict 
meta_model.fit(X_meta, y_real)
y_pred_stack = meta_model.predict(X_meta)

# Output fused predictions
df_fused = pd.DataFrame({
    "y_real": y_real,
    "y_pred_Base1": y_pred_lgbr,
    "y_pred_Base2": y_pred_sgb,
    "y_pred_Stacking": y_pred_stack
})

print("--- Sample Predictions ---")
print(df_fused.head())

# =====================================================================
# 3. GENERATE METRICS & EXPORT
# =====================================================================

print("\nGenerating Regression Reports...")
df_metrics = build_regression_reports(y_real, y_pred_stack)

print("\n--- Metrics Summary ---")
print(df_metrics)

# Exporting everything neatly into an Excel file
# output_excel = "Stacking_Regression_Report.xlsx"
# with pd.ExcelWriter(output_excel) as writer:
#     df_fused.to_excel(writer, sheet_name="Predictions", index=False)
    # df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
df_fused.to_clipboard(index=False)
# print(f"\nAll data successfully exported to {output_excel}")