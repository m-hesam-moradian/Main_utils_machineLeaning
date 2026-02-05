import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler


def shap_analysis_hr(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    save_path=None,
    sheet_name="SHAP_Sensitivity_HR",
):
    """
    Perform SHAP analysis specifically for Huber Regression.
    """
    # Ensure model is fitted
    try:
        model.predict(X_train)
    except:
        model.fit(X_train, y_train)

    # For Linear Models like Huber, shap.Explainer automatically 
    # selects the LinearExplainer logic.
    explainer = shap.Explainer(model, X_train, feature_names=X_train.columns)
    shap_values = explainer(X_test)

    # Extract SHAP values (for linear models, this is a 2D array: samples x features)
    # We use .values to get the matrix
    feature_names = X_test.columns.tolist()
    
    # Calculate Sensitivity (Global Importance)
    # Absolute mean of SHAP values per feature
    sensitivity_df = pd.DataFrame({
        "Feature": feature_names,
        "Sensitivity": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="Sensitivity", ascending=False).reset_index(drop=True)

    if save_path:
        with pd.ExcelWriter(save_path, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            sensitivity_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return sensitivity_df, shap_values

# --- Load dataset ---
sheet_name = "Data_after_KFold_GBR" # Updated to your current sheet
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()
target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# --- Scaling (Crucial for Huber SHAP interpretation) ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# --- Define and Fit HR Model ---
hr_model = HuberRegressor(epsilon=1.35, alpha=0.0001)
hr_model.fit(X_train, y_train)

# --- Run SHAP Analysis ---
sensitivity_df_shap, shap_values_obj = shap_analysis_hr(
    model=hr_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# --- Export to clipboard ---
print(sensitivity_df_shap)
sensitivity_df_shap.to_clipboard(index=False)