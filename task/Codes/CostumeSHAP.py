import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

# -------------------- 1. Load dataset --------------------
file_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.21-Data.xlsx"
sheet_name = "Data_after_KFold_LGBR"
target_column = "SOH"

df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()
X_raw = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
y = df[target_column].astype(float)

# -------------------- 2. Preprocess --------------------
X = StandardScaler().fit_transform(X_raw)
X = pd.DataFrame(X, columns=X_raw.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -------------------- 3. SHAP Analysis Function --------------------
def shap_analysis(model, X_train, y_train, X_test, y_test, save_path=None, sheet_name="SHAP_Sensitivity"):
    model.fit(X_train, y_train)
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Sensitivity scores
    sensitivity_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Sensitivity": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="Sensitivity", ascending=False).reset_index(drop=True)

    # Save to Excel if requested
    if save_path:
        with pd.ExcelWriter(save_path, engine="openpyxl", mode="a") as writer:
            sensitivity_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return sensitivity_df, shap_values

# -------------------- 4. Run SHAP for LGBR --------------------
model_lgbr = LGBMRegressor()
sensitivity_df_lgbr, shap_values_lgbr = shap_analysis(model_lgbr, X_train, y_train, X_test, y_test)

# -------------------- 5. Run SHAP for SGB --------------------
model_sgb = GradientBoostingRegressor()
sensitivity_df_sgb, shap_values_sgb = shap_analysis(model_sgb, X_train, y_train, X_test, y_test)

# -------------------- 6. Dependence Plots --------------------
top_features = sensitivity_df_lgbr["Feature"].head(3)  # Top 3 features from LGBR

for feature in top_features:
    shap.plots.scatter(shap_values_lgbr[:, feature], color=shap_values_lgbr)
    plt.title(f"LGBR SHAP Dependence: {feature}")
    plt.tight_layout()
    plt.show()

# -------------------- 7. Output --------------------
sensitivity_df_lgbr.to_clipboard(index=False)
# Optional: sensitivity_df_sgb.to_clipboard(index=False)