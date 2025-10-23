import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def shap_analysis(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    save_path=None,
    sheet_name="SHAP_Sensitivity",
):
    """
    Perform SHAP analysis for any trained model.
    """
    if not hasattr(model, "fit"):
        raise ValueError("Provided model is not a valid scikit-learn compatible model.")

    try:
        model.predict(X_train)
    except:
        model.fit(X_train, y_train)

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model, feature_names=X_train.columns)
    shap_values = explainer.shap_values(X_test)

    # Handle classification (binary or multiclass)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use class 1 for binary classification

    feature_names = (
        X_test.columns
        if hasattr(X_test, "columns")
        else [f"Feature_{i}" for i in range(X_test.shape[1])]
    )
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df["BaseValue"] = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    shap_df["ModelPrediction"] = shap_df[feature_names].sum(axis=1) + shap_df["BaseValue"]

    sensitivity_df = (
        pd.DataFrame({
            "Feature": feature_names,
            "Sensitivity": shap_df[feature_names].abs().mean(),
        })
        .sort_values(by="Sensitivity", ascending=False)
        .reset_index(drop=True)
    )

    if save_path:
        from openpyxl import load_workbook

        try:
            book = load_workbook(save_path)
            if sheet_name in book.sheetnames:
                book.remove(book[sheet_name])
                book.save(save_path)
        except FileNotFoundError:
            pass
        with pd.ExcelWriter(save_path, engine="openpyxl", mode="a" if save_path else "w") as writer:
            sensitivity_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return sensitivity_df, shap_values

# --- Load dataset ---
sheet_name = "Data_after_KFold"
file_path = r"C:\Users\Sam\Desktop\ML\task\BSS.No.1-Dataset.xlsx"
target_column = "Anomalous Load"

df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Run SHAP analysis
sensitivity_df_shap, shap_values = shap_analysis(
    model=RandomForestClassifier(),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# Copy results to clipboard
sensitivity_df_shap.to_clipboard(index=False)