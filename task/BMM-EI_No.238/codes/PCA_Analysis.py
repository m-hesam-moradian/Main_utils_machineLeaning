"""
Step 3 - PCA Analysis + Plots
=============================
Fit PCA on TRAINING data only (per task spec).
Retain components for 90%, 95%, 99% cumulative explained variance.
Produce:
  - Cumulative Explained Variance Plot
  - Scree Plot
  - PCA Loading Plot
  - PCA-transformed datasets (PCA90, PCA95, PCA99) for train + test
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA = r"/home/z/my-project/task_238/Data.xlsx"
PLOT_DIR = r"/home/z/my-project/task_238/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load train/test
df_train = pd.read_excel(DATA, sheet_name="Train_Data")
df_test  = pd.read_excel(DATA, sheet_name="Test_Data")
target_col = df_train.columns[-1]

X_train = df_train.drop(columns=[target_col]).values
y_train = df_train[target_col].values
X_test  = df_test.drop(columns=[target_col]).values
y_test  = df_test[target_col].values
feature_names = df_train.drop(columns=[target_col]).columns.tolist()

# Standardize (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Fit PCA on training data
print("Fitting PCA on training data ...")
pca_full = PCA()
pca_full.fit(X_train_scaled)
n_features = X_train_scaled.shape[1]

explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# Find number of components for each threshold
thresholds = {"90": 0.90, "95": 0.95, "99": 0.99}
n_components = {}
for label, thresh in thresholds.items():
    n_comp = int(np.searchsorted(cumulative_var, thresh) + 1)
    n_components[label] = n_comp
    print(f"  PCA {label}%: {n_comp} components (cumulative var = {cumulative_var[n_comp-1]:.4f})")

# ==================================================================
# Plot 1: Cumulative Explained Variance
# ==================================================================
print("\nGenerating Cumulative Explained Variance plot ...")
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.plot(range(1, n_features+1), cumulative_var, 'bo-', markersize=4, linewidth=1.5)
ax.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='90% variance')
ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% variance')
ax.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='99% variance')
for label, n_c in n_components.items():
    ax.axvline(x=n_c, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'{label}%\n({n_c} comp)', xy=(n_c, cumulative_var[n_c-1]),
                xytext=(n_c+1, cumulative_var[n_c-1]-0.08),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('PCA Cumulative Explained Variance')
ax.legend(loc='lower right')
ax.set_xticks(range(1, n_features+1))
ax.set_xticklabels(range(1, n_features+1), fontsize=7, rotation=45)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(PLOT_DIR, 'PCA_Cumulative_Variance.png'), dpi=300)
plt.close()
print("  [+] Saved PCA_Cumulative_Variance.png")

# ==================================================================
# Plot 2: Scree Plot
# ==================================================================
print("Generating Scree plot ...")
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.bar(range(1, n_features+1), explained_var, alpha=0.7, color='steelblue',
       label='Individual explained variance')
ax.plot(range(1, n_features+1), explained_var, 'ro-', markersize=4)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('PCA Scree Plot')
ax.set_xticks(range(1, n_features+1))
ax.set_xticklabels([f'PC{i}' for i in range(1, n_features+1)], fontsize=7, rotation=45)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(PLOT_DIR, 'PCA_Scree_Plot.png'), dpi=300)
plt.close()
print("  [+] Saved PCA_Scree_Plot.png")

# ==================================================================
# Plot 3: PCA Loading Plot (top 2 PCs)
# ==================================================================
print("Generating PCA Loading Plot ...")
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
loadings = pca_full.components_[:2].T  # shape (n_features, 2)
for i, feat in enumerate(feature_names):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
             color='steelblue', alpha=0.7, head_width=0.02, head_length=0.02)
    ax.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feat,
            fontsize=7, ha='center', va='center')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax.add_patch(circle)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
ax.set_title('PCA Loading Plot (PC1 vs PC2)')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
fig.savefig(os.path.join(PLOT_DIR, 'PCA_Loading_Plot.png'), dpi=300)
plt.close()
print("  [+] Saved PCA_Loading_Plot.png")

# ==================================================================
# Save PCA-transformed datasets
# ==================================================================
print("\nSaving PCA-transformed datasets ...")
pca_report = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(n_features)],
    "Explained_Variance": explained_var,
    "Cumulative_Variance": cumulative_var,
})

with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                     if_sheet_exists="replace") as writer:
    pca_report.to_excel(writer, sheet_name="PCA_Report", index=False)

    for label, n_c in n_components.items():
        pca = PCA(n_components=n_c)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca  = pca.transform(X_test_scaled)

        # Save train
        cols = [f"PC{i+1}" for i in range(n_c)] + [target_col]
        df_train_pca = pd.DataFrame(
            np.column_stack([X_train_pca, y_train]), columns=cols)
        df_test_pca = pd.DataFrame(
            np.column_stack([X_test_pca, y_test]), columns=cols)

        df_train_pca.to_excel(writer, sheet_name=f"Train_PCA{label}", index=False)
        df_test_pca.to_excel(writer, sheet_name=f"Test_PCA{label}", index=False)
        print(f"  PCA{label}: train={df_train_pca.shape}, test={df_test_pca.shape}")

print(f"\n[+] PCA analysis complete.")
print(f"[+] Plots saved to: {PLOT_DIR}")
