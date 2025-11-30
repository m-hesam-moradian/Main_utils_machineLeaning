"""
SHAP + SHAP-Uncertainty pipeline
- Keeps VIF-based feature selection (threshold 5)
- Trains predictive model (XGBoost by default) with RandomizedSearchCV
- Extracts hyperparameters from all search iterations
- Computes metrics: MAE, RMSE, R2, GR100, GR125
- Trains 'error model' on absolute errors and performs SHAP uncertainty analysis
- Creates SHAP plots similar to Figures 6,7,9a,11,12,13 from the uploaded ref
Requirements:
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn statsmodels openpyxl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from collections import defaultdict

# -------------------- USER SETTINGS --------------------
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Original old results\NBA_XGBR_DTR_ADAR_AFT_CSA_MOA_HGSO_En.xlsx"   # path to your data (change as needed)
SHEET_NAME = "VIF_DATA"                            # sheet name (if excel)
TARGET_COLUMN = None   # if None the script will choose the last column as target (matches your example)
TEST_SIZE = 0.2
RANDOM_STATE = 42
VIF_THRESHOLD = 5.0   # keep features with VIF <= 5 (scenario 1)
RESULTS_DIR = "shap_uncertainty_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# GR thresholds (assumption: percentage of predictions with |error| <= threshold)
GR_THRESHOLDS = [100, 125]  # GR100, GR125

# -------------------- 0. Load data --------------------
if DATA_PATH.lower().endswith(".xlsx") or DATA_PATH.lower().endswith(".xls"):
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME).dropna()
else:
    df = pd.read_csv(DATA_PATH).dropna()

if TARGET_COLUMN is None:
    TARGET_COLUMN = df.columns[-1]

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(float)

# -------------------- 1. VIF-based feature selection (iterative) --------------------
def calculate_vif(df_X):
    X_with_const = sm.add_constant(df_X)
    vif_data = pd.DataFrame({
        "feature": df_X.columns,
        "VIF": [variance_inflation_factor(X_with_const.values, i+1) for i in range(len(df_X.columns))]
    })
    return vif_data

X_vif = X.copy()
iteration = 0
removed_features = []
while True:
    vif_df = calculate_vif(X_vif)
    max_vif = vif_df["VIF"].max()
    if max_vif <= VIF_THRESHOLD:
        break
    # remove the feature with highest VIF
    to_remove = vif_df.sort_values("VIF", ascending=False).iloc[0]["feature"]
    removed_features.append((iteration, to_remove, float(max_vif)))
    X_vif = X_vif.drop(columns=[to_remove])
    iteration += 1

selected_features = X_vif.columns.tolist()
print(f"Selected {len(selected_features)} features after VIF pruning. Removed: {len(removed_features)} features.")
pd.DataFrame(removed_features, columns=["iter","feature_removed","vif_value"]).to_csv(os.path.join(RESULTS_DIR,"vif_removed_log.csv"), index=False)

# Use selected features only
X_sel = X[selected_features]

# -------------------- 2. Train/test split + scaling --------------------
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# -------------------- 3. Model + hyperparameter search --------------------
# Define model and param grid (adjust as you like)
model = XGBRegressor(objective="reg:squarederror", n_jobs=8, random_state=RANDOM_STATE, verbosity=0)

param_dist = {
    "n_estimators": [100, 200, 400, 800],
    "max_depth": [3, 5, 8, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0]
}

n_iter_search = 40
rsearch = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=RANDOM_STATE,
    return_train_score=False,
    verbose=1
)

rsearch.fit(X_train_scaled, y_train)

# Save the search object and extract params per iteration
joblib.dump(rsearch, os.path.join(RESULTS_DIR,"random_search.pkl"))
cv_results = pd.DataFrame(rsearch.cv_results_)
# Extract hyperparameters for each tried candidate
params_iter = pd.DataFrame(cv_results[['params','mean_test_score','rank_test_score']])
params_iter.to_csv(os.path.join(RESULTS_DIR,"hyperparameter_iterations.csv"), index=False)

best_model = rsearch.best_estimator_
print("Best params:", rsearch.best_params_)

# Save best model
joblib.dump(best_model, os.path.join(RESULTS_DIR,"best_model.pkl"))

# -------------------- 4. Predictions & metrics --------------------
y_pred = best_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
abs_err = np.abs(y_test - y_pred)

# GR100 / GR125 (assumption: proportion of test samples with abs error <= threshold)
gr_results = {}
for t in GR_THRESHOLDS:
    gr_results[f"GR{t}"] = 100.0 * (abs_err <= t).mean()

metrics_df = pd.DataFrame([{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    **gr_results
}])
metrics_df.to_csv(os.path.join(RESULTS_DIR,"metrics_summary.csv"), index=False)
print("Metrics:", metrics_df.to_dict(orient='records')[0])

# -------------------- 5. SHAP: explain main model (value predictions) --------------------
explainer = shap.TreeExplainer(best_model, data=X_train_scaled, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test_scaled)  # shape (n_samples, n_features) for tree models

# Mean absolute SHAP (Figure 6-like)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
imp_df = pd.DataFrame({"feature": X_train_scaled.columns, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
imp_df.to_csv(os.path.join(RESULTS_DIR,"shap_feature_importance.csv"), index=False)

plt.figure(figsize=(10,6))
plt.title("SHAP Summary Plot (Mean Absolute Values) - Model predictions")
plt.barh(imp_df['feature'].values[::-1], imp_df['mean_abs_shap'].values[::-1])
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"fig6_mean_abs_shap.png"), dpi=200)
plt.close()

# Full summary plot (dot plot) - top N
top_n = min(30, X_train_scaled.shape[1])
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, plot_type="dot", max_display=top_n, show=False)
plt.savefig(os.path.join(RESULTS_DIR,"fig6_summary_dot.png"), dpi=200, bbox_inches="tight")
plt.close()

# -------------------- 6. SHAP dependence plots for selected features (Figure 7-like) --------------------
# pick top features by mean_abs_shap
top_feats = imp_df['feature'].head(6).tolist()
for feat in top_feats:
    plt.figure(figsize=(6,4))
    try:
        shap.dependence_plot(feat, shap_values, X_test_scaled, show=False)
    except Exception as e:
        # fallback: scatter plot of feature values vs. SHAP
        print(f"dependence_plot failed for {feat}, using fallback scatter. Error: {e}")
        feat_idx = list(X_test_scaled.columns).index(feat)
        plt.scatter(X_test_scaled[feat], shap_values[:, feat_idx], alpha=0.5)
        plt.xlabel(feat)
        plt.ylabel("SHAP value")
    plt.title(f"Dependence plot for {feat}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,f"fig7_dependence_{feat}.png"), dpi=200)
    plt.close()

# -------------------- 7. SHAP interaction (Figure 9a-like) --------------------
# Many SHAP builds in your environment don't support interaction API.
# We'll compute a safe pseudo-interaction using correlation between SHAP columns.

print("Computing pseudo-interactions (correlation-based) for value SHAP...")

# Build DataFrame of SHAP values
shap_df = pd.DataFrame(shap_values, columns=X_test_scaled.columns)

# Correlation matrix of SHAP values (absolute correlation)
cor_matrix = shap_df.corr().abs()
np.fill_diagonal(cor_matrix.values, 0)  # ignore self-correlation

# Find top correlated pair (pseudo-interaction)
if cor_matrix.size == 0:
    print("No features to compute interactions.")
    top_pair = None
else:
    top_idx = np.unravel_index(np.argmax(cor_matrix.values), cor_matrix.shape)
    a = cor_matrix.index[top_idx[0]]
    b = cor_matrix.columns[top_idx[1]]
    print("Top pseudo-interaction pair (value SHAP):", a, b)

    # Scatter of SHAP(a) vs SHAP(b) as interaction-like figure
    plt.figure(figsize=(6,5))
    plt.scatter(shap_df[a], shap_df[b], alpha=0.5)
    plt.xlabel(f"SHAP for {a}")
    plt.ylabel(f"SHAP for {b}")
    plt.title(f"Pseudo SHAP interaction (correlation-based): {a} vs {b}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig9a_interaction_pseudo.png"), dpi=200)
    plt.close()

# -------------------- 8. SHAP-UNCERTAINTY: train model on absolute errors --------------------
# Prepare dataset for error modelling (following GeoVisX: recursing errors)
err = np.abs(y_test - y_pred)  # absolute errors on test; reference used this approach
# Use X_test_scaled as samples for error-model; split again internally
X_err = X_test_scaled.copy()
y_err = err.values

Xerr_train, Xerr_test, yerr_train, yerr_test = train_test_split(X_err, y_err, test_size=0.2, random_state=RANDOM_STATE)

# Use same model type and search (or simpler small search) - to save time we'll reuse best params but re-train
error_model = XGBRegressor(**rsearch.best_params_, objective="reg:squarederror", random_state=RANDOM_STATE)
error_model.fit(Xerr_train, yerr_train)

# SHAP for error model
explainer_err = shap.TreeExplainer(error_model, data=Xerr_train, feature_perturbation="interventional")
shap_values_err = explainer_err.shap_values(Xerr_test)

# Figure 11-like: variable ranking for uncertainty
mean_abs_shap_err = np.abs(shap_values_err).mean(axis=0)
imp_err_df = pd.DataFrame({"feature": Xerr_train.columns, "mean_abs_shap_err": mean_abs_shap_err}).sort_values("mean_abs_shap_err", ascending=False)
imp_err_df.to_csv(os.path.join(RESULTS_DIR,"shap_uncertainty_importance.csv"), index=False)

plt.figure(figsize=(10,6))
plt.title("SHAP (uncertainty) - Mean Absolute Values")
plt.barh(imp_err_df['feature'].values[::-1], imp_err_df['mean_abs_shap_err'].values[::-1])
plt.xlabel("Mean |SHAP value| for error model")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"fig11_uncertainty_mean_abs_shap.png"), dpi=200)
plt.close()

# proportion of samples where feature reduces error (SHAP < 0 reduces error)
prop_reduces = {}
for i,feat in enumerate(Xerr_test.columns):
    prop_reduces[feat] = (shap_values_err[:,i] < 0).mean()

prop_df = pd.DataFrame({"feature": list(prop_reduces.keys()), "prop_reduces_error": list(prop_reduces.values())}).sort_values("prop_reduces_error", ascending=False)
prop_df.to_csv(os.path.join(RESULTS_DIR,"feature_reduce_error_proportion.csv"), index=False)

plt.figure(figsize=(10,6))
sns.barplot(x="prop_reduces_error", y="feature", data=prop_df.head(30))
plt.xlabel("Proportion of samples where feature reduces error (SHAP < 0)")
plt.title("Figure 11b-like: features that most often reduce uncertainty")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"fig11b_proportion_reduces_error.png"), dpi=200)
plt.close()

# -------------------- 9. Marginal relationship plots for uncertainty (Figure 12-like) --------------------
top_err_feats = imp_err_df['feature'].head(12).tolist()
for feat in top_err_feats:
    plt.figure(figsize=(6,4))
    try:
        shap.dependence_plot(feat, shap_values_err, Xerr_test, show=False)
    except Exception as e:
        # fallback: scatter of feature values vs. SHAP-error
        print(f"dependence_plot (error) failed for {feat}, using fallback scatter. Error: {e}")
        feat_idx = list(Xerr_test.columns).index(feat)
        plt.scatter(Xerr_test[feat], shap_values_err[:, feat_idx], alpha=0.5)
        plt.xlabel(feat)
        plt.ylabel("SHAP (error) value")
    plt.title(f"Uncertainty dependence: {feat}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,f"fig12_uncert_dependence_{feat}.png"), dpi=200)
    plt.close()

# -------------------- 10. Interaction plots for uncertainty (Figure 13-like) --------------------
# Many SHAP builds don't support explainer.shap_interaction_values; use pseudo-interaction (correlation) instead.

print("Computing pseudo-interactions for uncertainty...")

shap_err_df = pd.DataFrame(shap_values_err, columns=Xerr_test.columns)
cor_err = shap_err_df.corr().abs()
np.fill_diagonal(cor_err.values, 0)

if cor_err.size == 0:
    print("No uncertainty features to compute interactions.")
else:
    top_idx_err = np.unravel_index(np.argmax(cor_err.values), cor_err.shape)
    a_err = cor_err.index[top_idx_err[0]]
    b_err = cor_err.columns[top_idx_err[1]]
    print("Top pseudo-interaction pair (uncertainty):", a_err, b_err)

    plt.figure(figsize=(6,5))
    plt.scatter(shap_err_df[a_err], shap_err_df[b_err], alpha=0.5)
    plt.xlabel(f"SHAP-error for {a_err}")
    plt.ylabel(f"SHAP-error for {b_err}")
    plt.title(f"Pseudo SHAP Uncertainty Interaction: {a_err} vs {b_err}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig13_uncert_interaction_pseudo.png"), dpi=200)
    plt.close()

# -------------------- 11. Save key tables for user --------------------
imp_df.to_csv(os.path.join(RESULTS_DIR,"value_shap_importance.csv"), index=False)
imp_err_df.to_csv(os.path.join(RESULTS_DIR,"uncertainty_shap_importance.csv"), index=False)
metrics_df.to_csv(os.path.join(RESULTS_DIR,"metrics_summary.csv"), index=False)

# --- Summary output to console ---
print("\n--- SUMMARY ---")
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
for k,v in gr_results.items():
    print(f"{k}: {v:.2f}%")
print(f"Saved plots & CSVs to folder: {RESULTS_DIR}")
