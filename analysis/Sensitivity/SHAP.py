import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")


def shap_analysis_rfc(
    model,
    X_train,
    X_test,
    save_path=None,
    sheet_name="SHAP_Sensitivity_RFC",
):
    """
    Perform SHAP sensitivity analysis for RandomForestClassifier.
    Works for binary and multi-class classification.
    """

    # TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer(X_test)

    feature_names = X_test.columns.tolist()

    # ---------------------------------------------------------
    # Handle different SHAP output shapes
    # ---------------------------------------------------------
    # Regression shape: (samples, features)
    # Classification shape: (samples, features, classes)

    shap_array = shap_values.values

    if len(shap_array.shape) == 3:
        # Multi-class or binary classifier
        # Take mean absolute over samples and classes
        shap_importance = np.abs(shap_array).mean(axis=(0, 2))
    else:
        # Regression case (not used here but safe)
        shap_importance = np.abs(shap_array).mean(axis=0)

    # Create Sensitivity DataFrame
    sensitivity_df = pd.DataFrame({
        "Feature": feature_names,
        "Sensitivity": shap_importance
    }).sort_values(by="Sensitivity", ascending=False).reset_index(drop=True)

    # Optional Excel saving
    if save_path:
        with pd.ExcelWriter(save_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            sensitivity_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return sensitivity_df, shap_values


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

sheet_name = "Data_after_KFold_RFC"
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()

target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# ---------------------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ---------------------------------------------------------
# TRAIN RANDOM FOREST CLASSIFIER
# ---------------------------------------------------------

rfc_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rfc_model.fit(X_train, y_train)

# ---------------------------------------------------------
# RUN SHAP ANALYSIS
# ---------------------------------------------------------

sensitivity_df_shap, shap_values_obj = shap_analysis_rfc(
    model=rfc_model,
    X_train=X_train,
    X_test=X_test,
    save_path=None  # Add Excel path here if needed
)

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

print("\nSHAP Sensitivity Results:\n")
print(sensitivity_df_shap)

# Copy to clipboard (optional)
sensitivity_df_shap.to_clipboard(index=False)

print("\nSuccess: Sensitivity results copied to clipboard.")
