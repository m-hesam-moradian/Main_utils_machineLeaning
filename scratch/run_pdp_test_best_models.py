# ==============================================================================
# MANDATORY RULE: PDP Sensitivity Analysis Guidelines
# 1. PDP is ALWAYS evaluated strictly on the holdout TEST partition (Test_Data),
#    NEVER on the training data.
# 2. PDP is ALWAYS based on the BEST performing model of each scenario
#    (e.g., Best Optimizer in Scenario 1 and Best Optimizer in Scenario 2).
# ==============================================================================

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor
from sklearn.inspection import partial_dependence
import openpyxl

excel_path = r"task/Data.xlsx"
bmm_excel_path = r"task/BMM-EI. No.227/Data.xlsx"

print("[*] Reading data from:", excel_path)

# 1. Get 24 ANOVA kept features
df_anova = pd.read_excel(excel_path, sheet_name="ANOVA_Report")
kept_features = df_anova[df_anova["Status"] == "Kept"]["Feature"].tolist()
print(f"[*] Total ANOVA Kept Features: {len(kept_features)}")

# 2. Read datasets
target_col = "Communication Reliability Score"

df_train_s1 = pd.read_excel(excel_path, sheet_name="Train_S1_With_Outlier")
df_train_s2 = pd.read_excel(excel_path, sheet_name="Train_S2_No_Outlier")
df_test = pd.read_excel(excel_path, sheet_name="Test_Data")

X_train_s1 = df_train_s1[kept_features].astype(float)
y_train_s1 = df_train_s1[target_col].astype(float)

X_train_s2 = df_train_s2[kept_features].astype(float)
y_train_s2 = df_train_s2[target_col].astype(float)

# RULE: Evaluated STRICTLY on Test_Data
X_test = df_test[kept_features].astype(float)
y_test = df_test[target_col].astype(float)

print(f"[*] Train S1 Shape: {X_train_s1.shape}")
print(f"[*] Train S2 Shape: {X_train_s2.shape}")
print(f"[*] Test Partition Shape: {X_test.shape} (strictly evaluated on holdout Test_Data)")

# 3. Fit Best Models for each Scenario
print("\n[*] Fitting Scenario 1 Best Model (QR + NOA)...")
# Tuned hyperparameters for Scenario 1 Champion (QR + NOA)
best_model_s1 = QuantileRegressor(alpha=0.004918, quantile=0.5, solver='highs')
best_model_s1.fit(X_train_s1, y_train_s1)
print(f"    S1 QR+NOA fitted. Train R2: {best_model_s1.score(X_train_s1, y_train_s1):.4f}, Test R2: {best_model_s1.score(X_test, y_test):.4f}")

print("\n[*] Fitting Scenario 2 Best Model (QR + NOA)...")
# Tuned hyperparameters for Scenario 2 Champion (QR + NOA)
best_model_s2 = QuantileRegressor(alpha=0.006282, quantile=0.5, solver='highs')
best_model_s2.fit(X_train_s2, y_train_s2)
print(f"    S2 QR+NOA fitted. Train R2: {best_model_s2.score(X_train_s2, y_train_s2):.4f}, Test R2: {best_model_s2.score(X_test, y_test):.4f}")

# 4. Directories for Plots
dirs = [
    r"task/PDP_outputs",
    r"task/BMM-EI. No.227/PDP_outputs",
    r"task/BMM-EI. No.227/PDP_outputs/Scenario_1_With_Outlier",
    r"task/BMM-EI. No.227/PDP_outputs/Scenario_2_Without_Outlier",
    r"task/BMM-EI. No.227/PDP_outputs/Comparative_S1_vs_S2"
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

# 5. Compute PDP curves on X_test for all 24 features
pdp_dict_s1 = {}
pdp_dict_s2 = {}

print("\n[*] Computing Partial Dependence on Test_Data for all 24 ANOVA features...")

for idx, feat in enumerate(kept_features, 1):
    # Scenario 1 (With Outlier Detection - Best Model: QR + NOA)
    pdp_s1 = partial_dependence(best_model_s1, X_test, [feat], grid_resolution=50)
    grid_s1 = pdp_s1["grid_values"][0]
    avg_s1 = pdp_s1["average"][0]
    pdp_dict_s1[f"{feat}_Grid"] = grid_s1
    pdp_dict_s1[f"{feat}_Partial_Dependence"] = avg_s1
    
    # Scenario 2 (Without Outlier Detection - Best Model: QR + NOA)
    pdp_s2 = partial_dependence(best_model_s2, X_test, [feat], grid_resolution=50)
    grid_s2 = pdp_s2["grid_values"][0]
    avg_s2 = pdp_s2["average"][0]
    pdp_dict_s2[f"{feat}_Grid"] = grid_s2
    pdp_dict_s2[f"{feat}_Partial_Dependence"] = avg_s2
    
    # Plot Scenario 1
    plt.figure(figsize=(6, 4))
    plt.plot(grid_s1, avg_s1, lw=2.2, color="#1f77b4", label="QR + NOA (Scenario 1)")
    plt.title(f"PDP (S1 With Outlier - Test): {feat}", fontsize=11, fontweight='bold')
    plt.xlabel(feat, fontsize=10)
    plt.ylabel("Partial Dependence", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(r"task/PDP_outputs", f"PDP_{feat}.png"), dpi=300)
    plt.savefig(os.path.join(r"task/BMM-EI. No.227/PDP_outputs", f"PDP_{feat}.png"), dpi=300)
    plt.savefig(os.path.join(r"task/BMM-EI. No.227/PDP_outputs/Scenario_1_With_Outlier", f"PDP_S1_{feat}.png"), dpi=300)
    plt.close()
    
    # Plot Scenario 2
    plt.figure(figsize=(6, 4))
    plt.plot(grid_s2, avg_s2, lw=2.2, color="#ff7f0e", label="QR + NOA (Scenario 2)")
    plt.title(f"PDP (S2 Without Outlier - Test): {feat}", fontsize=11, fontweight='bold')
    plt.xlabel(feat, fontsize=10)
    plt.ylabel("Partial Dependence", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(r"task/BMM-EI. No.227/PDP_outputs/Scenario_2_Without_Outlier", f"PDP_S2_{feat}.png"), dpi=300)
    plt.close()
    
    # Comparative Overlay Plot (S1 vs S2 on Test partition)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(grid_s1, avg_s1, lw=2.2, color="#1f77b4", label="S1: QR + NOA (With Outlier Detection)")
    plt.plot(grid_s2, avg_s2, lw=2.2, color="#e6550d", linestyle="--", label="S2: QR + NOA (Without Outlier Detection)")
    plt.title(f"PDP Comparison (Test Data): {feat}", fontsize=11, fontweight='bold')
    plt.xlabel(feat, fontsize=10)
    plt.ylabel("Partial Dependence (Reliability Score)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=9, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(r"task/BMM-EI. No.227/PDP_outputs/Comparative_S1_vs_S2", f"PDP_Comparison_{feat}.png"), dpi=300)
    plt.close()
    
    if idx % 6 == 0 or idx == len(kept_features):
        print(f"    Completed PDP & Plots for {idx}/{len(kept_features)} features...")

df_pdp_s1 = pd.DataFrame({k: pd.Series(v) for k, v in pdp_dict_s1.items()})
df_pdp_s2 = pd.DataFrame({k: pd.Series(v) for k, v in pdp_dict_s2.items()})

print("\n[*] Writing updated PDP Sensitivity sheets to Excel...")

def save_pdp_sheets(target_excel):
    with pd.ExcelWriter(target_excel, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        # Standard sheet name updated to full 24 features on Test partition
        df_pdp_s1.to_excel(writer, sheet_name="PDP_Sensitivity", index=False)
        df_pdp_s1.to_excel(writer, sheet_name="PDP_Sensitivity(With_Outlier)", index=False)
        df_pdp_s2.to_excel(writer, sheet_name="PDP_Sensitivity(Without_Outlier)", index=False)
    print(f"    Saved to: {target_excel}")

save_pdp_sheets(excel_path)
if os.path.exists(bmm_excel_path):
    save_pdp_sheets(bmm_excel_path)

print("\n[+] PDP Sensitivity Analysis strictly on Test_Data completed successfully!")
