import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import win32com.client

def close_excel_file(filepath):
    try:
        import time
        try:
            excel = win32com.client.GetActiveObject("Excel.Application")
            for wb in list(excel.Workbooks):
                try:
                    if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                        wb.Save()
                        wb.Close(SaveChanges=False)
                        print("[EXCEL] Saved and Closed Excel file:", filepath)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(0.5)
    except Exception as e:
        print("Note: Excel COM:", e)


def open_excel_file(filepath):
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[EXCEL] Opened Excel file:", filepath)
    except Exception:
        pass

# Load Excel data
file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "D4_Data"

print(f"Loading dataset for Decision Boundary extraction from sheet: '{sheet_name}'")
df = pd.read_excel(file_path, sheet_name=sheet_name)
target_column = df.columns[-1]
y = df[target_column].values
X = df.drop(columns=[target_column]).values

# Standardize and reduce to 2D using PCA for visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained Variance Ratio by 2 PCA components: {pca.explained_variance_ratio_}")

# Two hybrid models configured with Bayesian tuned hyperparameters
models = {
    "LR + Bayes": LogisticRegression(C=2.85, max_iter=300, random_state=42),
    "RFC + Bayes": RandomForestClassifier(n_estimators=75, max_depth=12, min_samples_split=3, random_state=42, n_jobs=-1)
}

# Create mesh grid for decision boundaries
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]

output_dir = r"C:\Users\Sam\Desktop\ML\task\Decision_Boundaries"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

boundary_summary = []

for idx, (m_name, clf) in enumerate(models.items()):
    print(f"Fitting {m_name} on 2D PCA projected space...")
    clf.fit(X_pca, y)
    Z = clf.predict(grid_points).reshape(xx.shape)
    
    ax = axes[idx]
    contour = ax.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=15, alpha=0.6)
    ax.set_title(f"Decision Boundary: {m_name} (D4)", fontsize=13, fontweight='bold')
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save individual high-res plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=15, alpha=0.6)
    plt.title(f"Decision Boundary - {m_name} (Dataset D4)", fontsize=14, fontweight='bold')
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    indiv_path = os.path.join(output_dir, f"Decision_Boundary_{m_name.replace(' + ', '_')}_D4.png")
    plt.savefig(indiv_path, bbox_inches='tight', dpi=300)
    plt.close()

    boundary_summary.append({
        "Model": m_name,
        "Dataset": "D4",
        "PC1_Variance_Ratio": pca.explained_variance_ratio_[0],
        "PC2_Variance_Ratio": pca.explained_variance_ratio_[1],
        "Total_2D_Variance_Explained": np.sum(pca.explained_variance_ratio_),
        "Boundary_Plot_File": os.path.basename(indiv_path)
    })

# Save combined plot
combined_path = os.path.join(output_dir, "Decision_Boundaries_Comparison_D4.png")
fig.tight_layout()
fig.savefig(combined_path, bbox_inches='tight', dpi=300)
plt.close(fig)

# Save summary to Excel
close_excel_file(file_path)
df_boundary_summary = pd.DataFrame(boundary_summary)
try:
    with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df_boundary_summary.to_excel(writer, sheet_name="Decision_Boundaries_D4", index=False)
    open_excel_file(file_path)
    print(f"[+] Summary saved to sheet 'Decision_Boundaries_D4' in {file_path}")
except PermissionError:
    print(f"[!] Note: task/Data.xlsx is currently open in Excel. Summary table will be written when file is saved.")
except Exception as e:
    print(f"Note: Excel write: {e}")

print("\n" + "="*70)
print(" DECISION BOUNDARY EXTRACTION COMPLETED (DATASET D4) ".center(70))
print("="*70)
print(df_boundary_summary.to_string(index=False))
print(f"\n[+] Plots saved in: {output_dir}")

