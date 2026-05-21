# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # --- Load reordered data for XGBR (after K-Fold) ---
# excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
# sheet_name = "Data_after_KFold_XGBoost"  # keep same sheet

# df = pd.read_excel(excel_path, sheet_name=sheet_name)
# target_column = df.columns[-1]

# X = df.drop(columns=target_column)
# y = df[target_column]

# # --- Use last 20% as test set to match K-Fold logic ---
# split_idx = int(len(df) * 0.8)
# X_train, X_test = X[:split_idx], X[split_idx:]
# y_train, y_test = y[:split_idx], y[split_idx:]

# # --- Define and train XGBR model ---
# model = XGBRegressor(
#     n_estimators=7,       # keep moderate
#     random_state=42
# )

# model.fit(X_train, y_train)

# # --- Predictions (Original) ---
# y_pred_all = model.predict(X)
# y_pred_train = model.predict(X_train)
# y_pred_test = model.predict(X_test)

# # --- Metrics ---
# mid = len(y_test) // 2
# sets = [
#     ("All", y, y_pred_all),
#     ("Train", y_train, y_pred_train),
#     ("Test", y_test, y_pred_test),
#     ("Value", y_test[:mid], y_pred_test[:mid]),
#     ("Test-Value", y_test[mid:], y_pred_test[mid:]),
# ]

# df_metrics = pd.DataFrame(
#     [
#         {
#             "Set": s,
#             "MAE": mean_absolute_error(y_t, y_p),
#             "RMSE": mean_squared_error(y_t, y_p) ** 0.5,
#             "R2": r2_score(y_t, y_p),
#         }
#         for s, y_t, y_p in sets
#     ]
# )

# print("--- Base Model Metrics ---")
# print(df_metrics)
# print("-" * 30)

# # =====================================================================
# # PROGRAMMATIC FUNCTION: Scenario A (Bounded Deterministic Perturbation)
# # =====================================================================
# def run_scenario_a(trained_model, X_data):
#     """
#     Scenario A (The Effort Simulation): 
#     Increase continuous effort metrics (Hours_Studied, Attendance) by +10% for all students.
#     """
#     # 1. Create a copy of the dataset to avoid overwriting the original training data
#     X_scenario = X_data.copy()
    
#     # 2. Apply the +10% rule (Multiplying by 1.10 adds 10%)
#     if 'Hours_Studied' in X_scenario.columns:
#         X_scenario['Hours_Studied'] = X_scenario['Hours_Studied'] * 1.10
        
#     if 'Attendance' in X_scenario.columns:
#         X_scenario['Attendance'] = X_scenario['Attendance'] * 1.10
        
#     # 3. Predict the target using the trained model on this new simulated data
#     scenario_predictions = trained_model.predict(X_scenario)
    
#     return scenario_predictions

# # --- Execute Scenario A on ALL students ---
# print("\nRunning Scenario A Simulation (+10% Effort)...")
# y_pred_scenario_a = run_scenario_a(model, X)

# # --- Output ONLY the new Scenario Predictions ---
# # If you want them rounded to integers like we discussed previously, you can uncomment the np.round()
# # y_pred_scenario_a = np.round(y_pred_scenario_a).astype(int)

# df_scenario_results = pd.DataFrame({
#     "Scenario_A_Predictions": y_pred_scenario_a
# })

# # --- Export to clipboard (Not the first prediction, ONLY the scenario predictions) ---
# df_scenario_results.to_clipboard(index=False, header=False)

# print("✅ Scenario A predictions computed!")
# print("✅ Copied to clipboard! Go to your 'Scenario A(DataSets)' sheet in Excel and press Paste.")

# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # =====================================================================
# # ACADEMIC ILLUSTRATION: Visualizing the Impact of Scenario A
# # =====================================================================
# print("\nGenerating academic figures for Scenario A...")

# # Create a folder to save the plots
# output_dir = "Scenario_Plots"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # We compare the BASE predictions to the SCENARIO predictions 
# # to isolate the exact impact of the +10% effort.
# base_preds = y_pred_all
# scenario_preds = y_pred_scenario_a

# # Set the visual style for academic papers
# sns.set_theme(style="whitegrid", context="paper")

# # ---------------------------------------------------------
# # Figure 1: Distribution Shift (Density Plot)
# # Shows how the overall "curve" of scores shifts upward
# # ---------------------------------------------------------
# plt.figure(figsize=(8, 5))
# sns.kdeplot(base_preds, fill=True, color="blue", label="Base Predictions (Original)", alpha=0.4)
# sns.kdeplot(scenario_preds, fill=True, color="green", label="Scenario A (+10% Effort)", alpha=0.4)
# plt.title("Impact of Increased Effort on Predicted Scores", fontsize=14, weight='bold')
# plt.xlabel("Predicted Score", fontsize=12)
# plt.ylabel("Density (Number of Students)", fontsize=12)
# plt.legend(loc="upper left")
# plt.savefig(os.path.join(output_dir, "Fig1_Distribution_Shift.png"), dpi=300, bbox_inches='tight')
# plt.close()

# # ---------------------------------------------------------
# # Figure 2: Scatter Plot with 1:1 Reference Line
# # Shows individual student score improvements
# # ---------------------------------------------------------
# plt.figure(figsize=(6, 6))
# plt.scatter(base_preds, scenario_preds, color="dodgerblue", alpha=0.6, edgecolor='k')

# # Draw the 1:1 Line (y=x). Points above this line mean an INCREASE in score.
# min_val = min(min(base_preds), min(scenario_preds)) - 2
# max_val = max(max(base_preds), max(scenario_preds)) + 2
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="No Change (y=x)")

# plt.title("Individual Student Shifts (Base vs Scenario A)", fontsize=14, weight='bold')
# plt.xlabel("Base Predicted Score", fontsize=12)
# plt.ylabel("Scenario A Predicted Score", fontsize=12)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.savefig(os.path.join(output_dir, "Fig2_Scatter_Shift.png"), dpi=300, bbox_inches='tight')
# plt.close()

# # ---------------------------------------------------------
# # Figure 3: Boxplot Comparison
# # Standard academic way to show Medians and Quartiles
# # ---------------------------------------------------------
# plt.figure(figsize=(6, 5))
# plot_data = pd.DataFrame({
#     "Base Predictions": base_preds,
#     "Scenario A (+10%)": scenario_preds
# })

# sns.boxplot(data=plot_data, palette=["#3498db", "#2ecc71"], width=0.5)
# plt.title("Statistical Summary of Score Improvement", fontsize=14, weight='bold')
# plt.ylabel("Predicted Score", fontsize=12)
# plt.savefig(os.path.join(output_dir, "Fig3_Boxplot_Comparison.png"), dpi=300, bbox_inches='tight')
# plt.close()

# # ---------------------------------------------------------
# # Print a quick statistical summary for your paper's text
# # ---------------------------------------------------------
# mean_increase = np.mean(scenario_preds - base_preds)
# max_increase = np.max(scenario_preds - base_preds)

# print(f"✅ Figures saved in folder: '{output_dir}'")
# print("\n--- Summary for your Paper's Text ---")
# print(f"By increasing student effort metrics by 10%, the model predicted an overall upward shift in scores.")
# print(f"The mean predicted score increased by {mean_increase:.2f} points.")
# print(f"The maximum individual score increase was {max_increase:.2f} points.")
# print("-" * 40)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# 1. SETUP & DATA LOADING
# =====================================================================

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data_after_KFold_XGBoost"

print("Loading data...")
try:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
except FileNotFoundError:
    print(f"Error: Could not find {excel_path}. Please check the path.")
    exit()

target_column = df.columns[-1]

X = df.drop(columns=target_column)
y = df[target_column]

# Split logic mapping K-Fold (using last 20% as test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Define and train base model
model = XGBRegressor( n_estimators=1000,       # keep moderate
    max_depth=3)
model.fit(X_train, y_train)

# Base Predictions (No rounding, kept as precise continuous variables)
y_pred_all = model.predict(X)

# =====================================================================
# 2. PROGRAMMATIC SCENARIOS (BOUNDED DETERMINISTIC PERTURBATION)
# =====================================================================

def run_scenario_a(trained_model, X_data):
    """
    Scenario A (+35% Effort): Increase ONLY Hours_Studied by 35%.
    """
    X_scenario = X_data.copy()
    features_altered = []
    
    if 'Hours_Studied' in X_scenario.columns:
        X_scenario['Hours_Studied'] *= 1.35  
        features_altered.append("Hours_Studied")
        
    return trained_model.predict(X_scenario), " & ".join(features_altered)

def run_scenario_b(trained_model, X_data):
    """
    Scenario B (-25% Resource Drop): Decrease Tutoring_Sessions and Internet_Access by 25%.
    """
    X_scenario = X_data.copy()
    features_altered = []
    
    if 'Tutoring_Sessions' in X_scenario.columns:
        X_scenario['Tutoring_Sessions'] *= 0.75  
        features_altered.append("Tutoring_Sessions")
        
    if 'Internet_Access' in X_scenario.columns:
        X_scenario['Internet_Access'] *= 0.75  
        features_altered.append("Internet_Access")
        
    return trained_model.predict(X_scenario), " & ".join(features_altered)

print("Running Scenario Simulations...")
y_pred_scenario_a, feat_A = run_scenario_a(model, X)
y_pred_scenario_b, feat_B = run_scenario_b(model, X)

# =====================================================================
# 3. METRICS CALCULATION (CAUSAL IMPACT)
# =====================================================================
PASS_THRESHOLD = 60

def calculate_causal_metrics(base_preds, scen_preds, feature_pct_change):
    base_mean = np.mean(base_preds)
    scen_mean = np.mean(scen_preds)
    
    # 1. Average Treatment Effect (ATE)
    ate = scen_mean - base_mean
    
    # 2. Feature Elasticity: (% Change in Score) / (% Change in Feature)
    pct_change_score = ate / base_mean
    elasticity = pct_change_score / feature_pct_change if feature_pct_change != 0 else np.nan
    
    # 3. Threshold Transition Rate
    if feature_pct_change > 0: # Positive intervention (rescued from failing)
        transition_count = np.sum((base_preds < PASS_THRESHOLD) & (scen_preds >= PASS_THRESHOLD))
    else: # Negative intervention (dropped to failing)
        transition_count = np.sum((base_preds >= PASS_THRESHOLD) & (scen_preds < PASS_THRESHOLD))
        
    return base_mean, scen_mean, ate, elasticity, transition_count

# Calculate for A (+35% -> 0.35)
base_mean, mean_A, ate_A, el_A, rescued_A = calculate_causal_metrics(y_pred_all, y_pred_scenario_a, 0.35)

# Calculate for B (-25% -> -0.25)
base_mean, mean_B, ate_B, el_B, failed_B = calculate_causal_metrics(y_pred_all, y_pred_scenario_b, -0.25)

# =====================================================================
# 4. EXPORTING REPORTS AND TABLES TO CSV
# =====================================================================

output_dir = "Scenario_Reports_and_Plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# A. Export Raw Predictions DataFrame
df_predictions = pd.DataFrame({
    "Student_ID": range(1, len(y_pred_all) + 1),
    "Baseline_Predicted_Score": y_pred_all,
    "Scenario_A_Predicted_Score": y_pred_scenario_a,
    "Scenario_B_Predicted_Score": y_pred_scenario_b,
    "Delta_Scenario_A": y_pred_scenario_a - y_pred_all,
    "Delta_Scenario_B": y_pred_scenario_b - y_pred_all
})

predictions_path = os.path.join(output_dir, "All_Predictions_and_Deltas.csv")
df_predictions.to_csv(predictions_path, index=False)

# B. Export The Scenario Impact Matrix (The Main Table)
matrix_data = [
    {
        "Scenario Name": "Scenario A: +35% Study Hours",
        "Feature Altered": feat_A if feat_A else "Hours_Studied",
        "Baseline Mean Score": round(base_mean, 2),
        "Scenario Mean Score": round(mean_A, 2),
        "ATE (Point Delta)": f"{ate_A:+.2f}",
        "Elasticity": round(el_A, 4),
        "Pass/Fail Transition": f"{rescued_A} Rescued"
    },
    {
        "Scenario Name": "Scenario B: -25% Tutoring & Internet",
        "Feature Altered": feat_B if feat_B else "Tutoring & Internet",
        "Baseline Mean Score": round(base_mean, 2),
        "Scenario Mean Score": round(mean_B, 2),
        "ATE (Point Delta)": f"{ate_B:+.2f}",
        "Elasticity": round(el_B, 4),
        "Pass/Fail Transition": f"{failed_B} Dropped to Fail"
    }
]

df_matrix = pd.DataFrame(matrix_data)
matrix_path = os.path.join(output_dir, "Scenario_Impact_Matrix.csv")
df_matrix.to_csv(matrix_path, index=False)

print("\n--- The Scenario Impact Matrix ---")
print(df_matrix.to_string(index=False))
print("----------------------------------\n")

# =====================================================================
# 5. HIGH-IMPACT ACADEMIC VISUALIZATIONS
# =====================================================================
print("Generating academic figures...")

sns.set_theme(style="whitegrid", context="paper")

# Define standard colors for the paper
COLOR_BASE = "blue"      # Blue
COLOR_SCEN_A = "green"   # Green
COLOR_SCEN_B = "red"     # Red

# ---------------------------------------------------------
# Figure 1: KDE Density Plot (The Specific Figure Requested)
# Replicates the "bell curve" filled areas showing distribution shifts
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

sns.kdeplot(y_pred_all, fill=True, color=COLOR_BASE, label="Base Predictions (Original)", alpha=0.4)
sns.kdeplot(y_pred_scenario_a, fill=True, color=COLOR_SCEN_A, label="Scenario A (+35% Study Hours)", alpha=0.4)
sns.kdeplot(y_pred_scenario_b, fill=True, color=COLOR_SCEN_B, label="Scenario B (-25% Tutoring & Internet)", alpha=0.4)

plt.title("Distribution Shift: Impact of Perturbation Scenarios on Predicted Scores", fontsize=15, weight='bold')
plt.xlabel("Predicted Score", fontsize=13)
plt.ylabel("Density (Number of Students)", fontsize=13)
plt.legend(loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Fig1_KDE_Distribution_Shift.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 2: Side-by-Side Violin Plots
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plot_data_violin = pd.DataFrame({
    "Baseline": y_pred_all,
    "Scenario A (+35%)": y_pred_scenario_a,
    "Scenario B (-25%)": y_pred_scenario_b
})

# Melt for seaborn
melted_violin = plot_data_violin.melt(var_name="Scenario", value_name="Score")

sns.violinplot(
    x="Scenario", y="Score", data=melted_violin, 
    palette=["#3498db", "#2ecc71", "#e74c3c"], inner="quartile"
)
plt.title("Statistical Quartile Shift: Baseline vs. Simulated Scenarios", fontsize=14, weight='bold')
plt.ylabel("Predicted Exam Score", fontsize=12)
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Fig2_ViolinPlot_Comparison.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 3: Scatter Plots with 45-Degree Equality Line
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

# Determine universal axis limits for a perfect 1:1 square look
min_val = min(np.min(y_pred_all), np.min(y_pred_scenario_a), np.min(y_pred_scenario_b)) - 5
max_val = max(np.max(y_pred_all), np.max(y_pred_scenario_a), np.max(y_pred_scenario_b)) + 5

# Subplot 1: Scenario A
axes[0].scatter(y_pred_all, y_pred_scenario_a, color="#2ecc71", alpha=0.6, edgecolor='k', label="Student Shift (+)")
axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label="Equality Line (No Impact)")
axes[0].set_title("Scenario A Impact (+35% Study Hours)", fontsize=12, weight='bold')
axes[0].set_xlabel("Baseline Predicted Score", fontsize=11)
axes[0].set_ylabel("Scenario Predicted Score", fontsize=11)
axes[0].legend(loc="upper left")

# Subplot 2: Scenario B
axes[1].scatter(y_pred_all, y_pred_scenario_b, color="#e74c3c", alpha=0.6, edgecolor='k', label="Student Shift (-)")
axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label="Equality Line (No Impact)")
axes[1].set_title("Scenario B Impact (-25% Tutoring & Internet)", fontsize=12, weight='bold')
axes[1].set_xlabel("Baseline Predicted Score", fontsize=11)
axes[1].legend(loc="lower right")

# Set global limits
plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Fig3_Scatter_45DegreeLine.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 4: Cumulative Distribution Function (CDF) Curve
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

# Plot ECDFs
sns.ecdfplot(y_pred_all, color="#3498db", label="Baseline Curve", linewidth=2.5)
sns.ecdfplot(y_pred_scenario_a, color="#2ecc71", label="Scenario A (+35%)", linewidth=2.5)
sns.ecdfplot(y_pred_scenario_b, color="#e74c3c", label="Scenario B (-25%)", linewidth=2.5)

# Vertical Pass/Fail Line
plt.axvline(x=PASS_THRESHOLD, color='black', linestyle=':', linewidth=2, label=f"Passing Grade ({PASS_THRESHOLD})")

# Visual Annotations
plt.title("Cumulative Distribution: Pass/Fail Threshold Shifts", fontsize=14, weight='bold')
plt.xlabel("Predicted Exam Score", fontsize=12)
plt.ylabel("Cumulative Proportion of Students", fontsize=12)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Fig4_CDF_PassFailImpact.png"), dpi=300)
plt.close()

print(f"✅ Execution Complete! All CSV files and Figures have been saved into the '{output_dir}' directory.")