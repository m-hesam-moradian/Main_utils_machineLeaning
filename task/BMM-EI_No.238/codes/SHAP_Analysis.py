"""
Step 6 - SHAP Analysis (Test Dataset Only)
==========================================
Run SHAP on test dataset for Condition A (original features).
Produce:
  - SHAP ranking of feature sensitivity (per model)
  - SHAP uncertainty comparison plots (across models)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA = r"/home/z/my-project/task_238/Data.xlsx"
PLOT_DIR = r"/home/z/my-project/task_238/plots"

# Load test data (Condition A = original features)
df_train = pd.read_excel(DATA, sheet_name="Train_Data")
df_test  = pd.read_excel(DATA, sheet_name="Test_Data")
target_col = df_train.columns[-1]
feature_names = df_train.drop(columns=[target_col]).columns.tolist()

X_train = df_train.drop(columns=[target_col]).values
y_train = df_train[target_col].values
X_test  = df_test.drop(columns=[target_col]).values
y_test  = df_test[target_col].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

MODELS = {
    "LR":   LogisticRegression(C=3.84719234, max_iter=500, tol=0.0001,
                                solver='lbfgs', random_state=42),
    "RFC":  RandomForestClassifier(n_estimators=287, max_depth=20,
                                    min_samples_split=2, min_samples_leaf=1,
                                    random_state=42, n_jobs=-1),
    "XGBC": XGBClassifier(n_estimators=287, max_depth=6, learning_rate=0.0482914,
                          subsample=0.85, colsample_bytree=0.85,
                          eval_metric='mlogloss', random_state=42),
}

# ------------------------------------------------------------------
# Compute mean abs SHAP per feature per model
# ------------------------------------------------------------------
shap_rankings = {}

for model_name, model in MODELS.items():
    print(f"\nRunning SHAP for {model_name} on test dataset ...")
    model.fit(X_train_s, y_train)

    if model_name in ["RFC", "XGBC"]:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test_s)
        # TreeExplainer for multiclass returns list of (n_samples, n_features)
        if isinstance(sv, list):
            # mean abs across classes then across samples
            mean_abs = np.mean([np.mean(np.abs(s), axis=0) for s in sv], axis=0)
        elif sv.ndim == 3:
            mean_abs = np.mean(np.abs(sv), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(sv), axis=0)
    else:
        explainer = shap.LinearExplainer(model, X_train_s)
        sv = explainer.shap_values(X_test_s)
        if isinstance(sv, list):
            mean_abs = np.mean([np.mean(np.abs(s), axis=0) for s in sv], axis=0)
        elif sv.ndim == 3:
            mean_abs = np.mean(np.abs(sv), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(sv), axis=0)

    shap_rankings[model_name] = mean_abs
    print(f"  Mean abs SHAP shape: {mean_abs.shape}")

    # SHAP bar plot (simple, no beeswarm to avoid layout issues)
    ranked_idx = np.argsort(mean_abs)[::-1]
    sorted_vals = mean_abs[ranked_idx]
    sorted_names = [feature_names[j] for j in ranked_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_names)), sorted_vals, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'SHAP Feature Importance - {model_name} (Test Data)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'SHAP_Summary_{model_name}.png'), dpi=300)
    plt.close()
    print(f"  [+] Saved SHAP_Summary_{model_name}.png")

# ------------------------------------------------------------------
# SHAP Ranking Table
# ------------------------------------------------------------------
print("\nBuilding SHAP ranking table ...")
ranking_rows = []
for model_name in MODELS:
    vals = shap_rankings[model_name]
    ranked_idx = np.argsort(vals)[::-1]
    for rank, idx in enumerate(ranked_idx, 1):
        ranking_rows.append({
            "Model": model_name, "Rank": rank,
            "Feature": feature_names[idx], "Mean_Abs_SHAP": vals[idx],
        })
df_shap_rank = pd.DataFrame(ranking_rows)

with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                     if_sheet_exists="replace") as writer:
    df_shap_rank.to_excel(writer, sheet_name="SHAP_Ranking", index=False)
print("[+] Saved SHAP_Ranking sheet")

# ------------------------------------------------------------------
# SHAP Uncertainty Comparison Plot
# ------------------------------------------------------------------
print("Generating SHAP uncertainty comparison plot ...")
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(feature_names))
width = 0.25
colors = ['#4C72B0', '#DD8452', '#55A467']
for i, model_name in enumerate(MODELS):
    vals = shap_rankings[model_name]
    vals_norm = vals / vals.sum() if vals.sum() > 0 else vals
    ax.bar(x + i * width, vals_norm, width, label=model_name, color=colors[i], alpha=0.8)
ax.set_xlabel('Feature')
ax.set_ylabel('Normalized SHAP Importance')
ax.set_title('SHAP Feature Sensitivity Comparison (Test Dataset)')
ax.set_xticks(x + width)
ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=7)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'SHAP_Uncertainty_Comparison.png'), dpi=300)
plt.close()
print("[+] Saved SHAP_Uncertainty_Comparison.png")

# ------------------------------------------------------------------
# SHAP Ranking Bar Plot (side by side)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for i, model_name in enumerate(MODELS):
    vals = shap_rankings[model_name]
    ranked_idx = np.argsort(vals)[::-1]
    sorted_vals = vals[ranked_idx]
    sorted_names = [feature_names[j] for j in ranked_idx]
    axes[i].barh(range(len(sorted_names)), sorted_vals, color=colors[i], alpha=0.8)
    axes[i].set_yticks(range(len(sorted_names)))
    axes[i].set_yticklabels(sorted_names, fontsize=7)
    axes[i].invert_yaxis()
    axes[i].set_xlabel('Mean |SHAP|')
    axes[i].set_title(f'{model_name}')
    axes[i].grid(True, alpha=0.3, axis='x')
fig.suptitle('SHAP Feature Ranking (Test Dataset)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'SHAP_Ranking_Bar.png'), dpi=300)
plt.close()
print("[+] Saved SHAP_Ranking_Bar.png")

print("\n[+] SHAP analysis complete.")
