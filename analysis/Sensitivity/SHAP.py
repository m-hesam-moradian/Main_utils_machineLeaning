import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor  # Changed model import
from sklearn.preprocessing import StandardScaler

def shap_analysis_lgbm(
    model,
    X_train,
    X_test,
    save_path=None,
    sheet_name="SHAP_Sensitivity_LGBM",
):
    """
    Perform SHAP analysis specifically for LightGBM.
    """
    # LightGBM uses TreeExplainer for high performance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    feature_names = X_test.columns.tolist()
    
    # Calculate Sensitivity (Global Importance)
    # Absolute mean of SHAP values per feature
    sensitivity_df = pd.DataFrame({
        "Feature": feature_names,
        "Sensitivity": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="Sensitivity", ascending=False).reset_index(drop=True)

    if save_path:
        # Note: 'a' mode requires the file to exist. Use 'w' if it's a new file.
        with pd.ExcelWriter(save_path, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            sensitivity_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return sensitivity_df, shap_values

# --- Load dataset ---
sheet_name = "Data_after_KFold_NGBM" 
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()
target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# --- Train-Test Split ---
# Note: Tree models like LGBM are scale-invariant, but keeping scaling won't hurt.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Define and Fit LGBM Model ---
lgbm_model = LGBMRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=-1, 
    random_state=42,
    verbose=-1 # Silences the output
)
lgbm_model.fit(X_train, y_train)

# --- Run SHAP Analysis ---
sensitivity_df_shap, shap_values_obj = shap_analysis_lgbm(
    model=lgbm_model,
    X_train=X_train,
    X_test=X_test,
)

# --- Output ---
print(sensitivity_df_shap)
sensitivity_df_shap.to_clipboard(index=False)
print("\nSuccess: Sensitivity results copied to clipboard.")