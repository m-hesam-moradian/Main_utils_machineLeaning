import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# =====================================================================
# 1. SETUP & DATA LOADING
# =====================================================================
excel_path = r"d:\ML\task\Data.xlsx"
sheet_name = "Optimal_Data_MRMR"

print("Loading data...")
try:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
except FileNotFoundError:
    print(f"Error: Could not find {excel_path}. Please check the path.")
    exit()

target_column = df.columns[-1]
X = df.drop(columns=target_column)
y = df[target_column]

# Define and train base model
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("Training base XGBRegressor model...")
model = XGBRegressor(n_estimators=1000, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Base Predictions
y_pred_all = model.predict(X)

output_dir = r"d:\ML\task\Scenario_Reports_and_Plots"
os.makedirs(output_dir, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

PASS_THRESHOLD = 60.0
MAX_HOURS_STUDIED = 37.0
MIN_RESOURCES = 0.0

failing_indices = np.where(y_pred_all < PASS_THRESHOLD)[0]
passing_indices = np.where(y_pred_all >= PASS_THRESHOLD)[0]
base_mean = np.mean(y_pred_all)

# =====================================================================
# SCENARIO A: POSITIVE EFFORT (HOURS STUDIED)
# =====================================================================
print("\n" + "="*50)
print("SCENARIO A: POSITIVE EFFORT (Phase II-A & Phase II-B)")
print("="*50)

# --- Phase II-A: Sensitivity Analysis (5% to 50% Increase) ---
sensitivity_steps_pos = np.arange(0.05, 0.55, 0.05)
results_phase_a_pos = []

for step in sensitivity_steps_pos:
    X_scenario = X.copy()
    X_scenario['Hours_Studied'] = np.minimum(X_scenario['Hours_Studied'] * (1 + step), MAX_HOURS_STUDIED)
    
    preds_scenario = model.predict(X_scenario)
    scen_mean = np.mean(preds_scenario)
    
    ate = scen_mean - base_mean
    pct_change_score = ate / base_mean
    elasticity = pct_change_score / step if step != 0 else np.nan
    rescued_count = np.sum((y_pred_all < PASS_THRESHOLD) & (preds_scenario >= PASS_THRESHOLD))
    
    results_phase_a_pos.append({
        'Intervention_Pct': step * 100,
        'ATE': ate,
        'Elasticity': elasticity,
        'Rescued_Students': rescued_count
    })

df_phase_a_pos = pd.DataFrame(results_phase_a_pos)
df_phase_a_pos.to_csv(os.path.join(output_dir, "ScenarioA_Sensitivity_Analysis.csv"), index=False)

# Plot Phase II-A (Positive)
fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:blue'
ax1.set_xlabel('Increase in Hours Studied (%)', fontsize=12, weight='bold')
ax1.set_ylabel('Average Treatment Effect (ATE)', color=color, fontsize=12, weight='bold')
ax1.plot(df_phase_a_pos['Intervention_Pct'], df_phase_a_pos['ATE'], marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Students Rescued (TTR)', color=color, fontsize=12, weight='bold')
ax2.plot(df_phase_a_pos['Intervention_Pct'], df_phase_a_pos['Rescued_Students'], marker='s', color=color, linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Scenario A: Sensitivity Analysis of Effort Intervention", fontsize=14, weight='bold')
plt.axvline(35, color='gray', linestyle=':', label='35% Benchmark (Diminishing Returns)')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, "ScenarioA_Sensitivity_Plot.svg"), format='svg', dpi=300)
plt.close()

# --- Phase II-B: Individualized Minimal Intervention (Positive) ---
increments_pos = np.arange(0.01, 1.01, 0.01)
X_failing_base = X.iloc[failing_indices].copy()
base_hours = X_failing_base['Hours_Studied'].values
required_increments_pos = np.full(len(failing_indices), np.nan)

for inc in increments_pos:
    not_rescued = np.isnan(required_increments_pos)
    if not np.any(not_rescued): break
    
    X_test_inc = X_failing_base.copy()
    X_test_inc['Hours_Studied'] = np.minimum(base_hours * (1 + inc), MAX_HOURS_STUDIED)
    preds_inc = model.predict(X_test_inc)
    
    newly_passed = (preds_inc >= PASS_THRESHOLD) & not_rescued
    required_increments_pos[newly_passed] = inc

rescued_mask = ~np.isnan(required_increments_pos)
rescued_increments_pos = required_increments_pos[rescued_mask]

# Export CSV for Individualized Interventions (Positive)
df_imi_pos = X_failing_base.copy()
df_imi_pos['Baseline_Predicted_Score'] = y_pred_all[failing_indices]
df_imi_pos['Required_Increase_Pct'] = required_increments_pos
df_imi_pos['Rescued'] = rescued_mask
df_imi_pos.to_csv(os.path.join(output_dir, "ScenarioA_Individualized_Interventions.csv"), index=False)

if len(rescued_increments_pos) > 0:
    plt.figure(figsize=(8, 5))
    sns.histplot(rescued_increments_pos * 100, bins=20, kde=True, color="green")
    plt.title("Distribution of Minimal Required Study Hours Increase\n(For Failing Students)", fontsize=14, weight='bold')
    plt.xlabel("Required Increase in Hours Studied (%)", fontsize=12)
    plt.ylabel("Number of Students", fontsize=12)
    plt.axvline(np.median(rescued_increments_pos)*100, color='r', linestyle='--', label=f'Median (+{np.median(rescued_increments_pos)*100:.1f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ScenarioA_Minimal_Increase_Dist.svg"), format='svg', dpi=300)
    plt.close()


# =====================================================================
# SCENARIO B: NEGATIVE RESOURCE DROP (TUTORING & INTERNET)
# =====================================================================
print("\n" + "="*50)
print("SCENARIO B: NEGATIVE RESOURCE DROP (Phase II-A & Phase II-B)")
print("="*50)

# --- Phase II-A: Sensitivity Analysis (5% to 50% Decrease) ---
sensitivity_steps_neg = np.arange(0.05, 0.55, 0.05)
results_phase_a_neg = []

for step in sensitivity_steps_neg:
    X_scenario = X.copy()
    if 'Tutoring_Sessions' in X_scenario.columns:
        X_scenario['Tutoring_Sessions'] = np.maximum(X_scenario['Tutoring_Sessions'] * (1 - step), MIN_RESOURCES)
    if 'Internet_Access' in X_scenario.columns:
        X_scenario['Internet_Access'] = np.maximum(X_scenario['Internet_Access'] * (1 - step), MIN_RESOURCES)
    
    preds_scenario = model.predict(X_scenario)
    scen_mean = np.mean(preds_scenario)
    
    ate = scen_mean - base_mean # will be negative
    pct_change_score = ate / base_mean
    elasticity = pct_change_score / (-step) if step != 0 else np.nan
    dropped_count = np.sum((y_pred_all >= PASS_THRESHOLD) & (preds_scenario < PASS_THRESHOLD))
    
    results_phase_a_neg.append({
        'Resource_Drop_Pct': step * 100,
        'ATE': ate,
        'Elasticity': elasticity,
        'Dropped_Students': dropped_count
    })

df_phase_a_neg = pd.DataFrame(results_phase_a_neg)
df_phase_a_neg.to_csv(os.path.join(output_dir, "ScenarioB_Sensitivity_Analysis.csv"), index=False)

# Plot Phase II-A (Negative)
fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:blue'
ax1.set_xlabel('Decrease in Resources (Tutoring & Internet %)', fontsize=12, weight='bold')
ax1.set_ylabel('Average Treatment Effect (ATE)', color=color, fontsize=12, weight='bold')
ax1.plot(df_phase_a_neg['Resource_Drop_Pct'], df_phase_a_neg['ATE'], marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Students Dropped to Fail (TTR)', color=color, fontsize=12, weight='bold')
ax2.plot(df_phase_a_neg['Resource_Drop_Pct'], df_phase_a_neg['Dropped_Students'], marker='s', color=color, linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Scenario B: Sensitivity Analysis of Resource Deprivation", fontsize=14, weight='bold')
plt.axvline(25, color='gray', linestyle=':', label='25% Benchmark')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, "ScenarioB_Sensitivity_Plot.svg"), format='svg', dpi=300)
plt.close()

# --- Phase II-B: Individualized Minimal Vulnerability (Negative) ---
# For students currently PASSING, what is the minimal % drop in resources needed to make them FAIL?
decrements_neg = np.arange(0.01, 1.01, 0.01)
X_passing_base = X.iloc[passing_indices].copy()
base_tutoring = X_passing_base.get('Tutoring_Sessions', np.zeros(len(passing_indices))).values
base_internet = X_passing_base.get('Internet_Access', np.zeros(len(passing_indices))).values

required_decrements_neg = np.full(len(passing_indices), np.nan)

for dec in decrements_neg:
    not_dropped = np.isnan(required_decrements_neg)
    if not np.any(not_dropped): break
    
    X_test_dec = X_passing_base.copy()
    if 'Tutoring_Sessions' in X_test_dec.columns:
        X_test_dec['Tutoring_Sessions'] = np.maximum(base_tutoring * (1 - dec), MIN_RESOURCES)
    if 'Internet_Access' in X_test_dec.columns:
        X_test_dec['Internet_Access'] = np.maximum(base_internet * (1 - dec), MIN_RESOURCES)
        
    preds_dec = model.predict(X_test_dec)
    
    newly_failed = (preds_dec < PASS_THRESHOLD) & not_dropped
    required_decrements_neg[newly_failed] = dec

dropped_mask = ~np.isnan(required_decrements_neg)
dropped_decrements_neg = required_decrements_neg[dropped_mask]

# Export CSV for Individualized Vulnerabilities (Negative)
df_imi_neg = X_passing_base.copy()
df_imi_neg['Baseline_Predicted_Score'] = y_pred_all[passing_indices]
df_imi_neg['Required_Resource_Drop_Pct'] = required_decrements_neg
df_imi_neg['Dropped_To_Fail'] = dropped_mask
df_imi_neg.to_csv(os.path.join(output_dir, "ScenarioB_Individualized_Vulnerabilities.csv"), index=False)

if len(dropped_decrements_neg) > 0:
    plt.figure(figsize=(8, 5))
    sns.histplot(dropped_decrements_neg * 100, bins=20, kde=True, color="red")
    plt.title("Distribution of Minimal Resource Drop to Induce Failure\n(For Vulnerable Passing Students)", fontsize=14, weight='bold')
    plt.xlabel("Required Decrease in Resources (%)", fontsize=12)
    plt.ylabel("Number of Vulnerable Students", fontsize=12)
    plt.axvline(np.median(dropped_decrements_neg)*100, color='black', linestyle='--', label=f'Median Drop (-{np.median(dropped_decrements_neg)*100:.1f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ScenarioB_Minimal_Drop_Dist.svg"), format='svg', dpi=300)
    plt.close()

print("\n[SUCCESS] Script completed successfully! All CSVs and SVG figures are saved in the output directory.")
