# =========================================================
# SHAP Dependence Plots (Matplotlib Style for SGB)
# =========================================================
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# -------------------- 1. Load dataset --------------------
file_path = r"C:\Users\Sam\Desktop\BMM-EI. No.21-Data.xlsx"
sheet_name = "Data_after_KFold_LGBR"
target_column = "SOH"

df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()
X_raw = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
y = df[target_column].astype(float)

# -------------------- 2. Preprocess --------------------
X = StandardScaler().fit_transform(X_raw)
X = pd.DataFrame(X, columns=X_raw.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -------------------- 3. SHAP Analysis --------------------
def shap_analysis(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    sensitivity_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Sensitivity": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="Sensitivity", ascending=False).reset_index(drop=True)

    return sensitivity_df, shap_values

# -------------------- 4. Train SGB model --------------------
model_sgb = GradientBoostingRegressor()
sensitivity_df_sgb, shap_values_sgb = shap_analysis(model_sgb, X_train, y_train, X_test)

# -------------------- 5. Matplotlib Dependence Plots --------------------
# Top 3 features by mean |SHAP|
top_features = sensitivity_df_sgb["Feature"].head(3).tolist()
print("Top 3 SGB features:", top_features)

plt.style.use("seaborn-v0_8-whitegrid")  # for SHAP-like style

for i, feature in enumerate(top_features):
    feature_idx = list(X_test.columns).index(feature)
    shap_feature_values = shap_values_sgb.values[:, feature_idx]
    feature_data = X_test.iloc[:, feature_idx]

    # Choose another feature for color (interaction)
    color_feature = top_features[(i + 1) % len(top_features)]
    color_idx = list(X_test.columns).index(color_feature)
    color_data = X_test.iloc[:, color_idx]

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        feature_data,
        shap_feature_values,
        c=color_data,
        cmap="coolwarm",
        s=30,
        alpha=0.8,
        edgecolors="none"
    )
    plt.colorbar(sc, label=f"{color_feature} value")
    plt.xlabel(feature, fontsize=11)
    plt.ylabel(f"SHAP value for {feature}", fontsize=11)
    plt.title(f"SGB SHAP Dependence: {feature} vs {color_feature}", fontsize=13, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# -------------------- 6. Output --------------------
sensitivity_df_sgb.to_clipboard(index=False)
print("\n✅ SGB sensitivity table copied to clipboard.")
