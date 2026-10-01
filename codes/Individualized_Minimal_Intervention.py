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
# We train on the same logic as XGBRwithScenario.py (first 80% for training)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("Training base XGBRegressor model...")
model = XGBRegressor(n_estimators=1000, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Base Predictions
y_pred_all = model.predict(X)

# =====================================================================
# 2. INDIVIDUALIZED MINIMAL INTERVENTION
# =====================================================================
PASS_THRESHOLD = 60.0
MAX_HOURS_STUDIED = 37.0  # From dataset properties
MAX_INCREASE_PCT = 1.0    # 100% max increase considered realistic/actionable

# Identify failing students
failing_indices = np.where(y_pred_all < PASS_THRESHOLD)[0]
print(f"Total students: {len(y_pred_all)}")
print(f"Failing students (Predicted < {PASS_THRESHOLD}): {len(failing_indices)}")

# To store the minimum required percentage increase for each failing student
min_req_increases = []
# To track students who cannot be rescued within the bounds
unrescuable_count = 0
unrescuable_indices = []

print("Running Individualized Minimal Intervention analysis...")

# We will test increments from 1% to 100% (0.01 to 1.00)
increments = np.arange(0.01, MAX_INCREASE_PCT + 0.01, 0.01)

# We can vectorize this to make it faster
# Create a copy of the features for failing students
X_failing_base = X.iloc[failing_indices].copy()
base_hours = X_failing_base['Hours_Studied'].values

required_increments = np.full(len(failing_indices), np.nan)

# Iterate through increments. Since predictions monotonically increase with Hours_Studied,
# we can just find the first increment that pushes the score >= 60.
for inc in increments:
    # Get indices of students not yet rescued
    not_rescued_mask = np.isnan(required_increments)
    if not np.any(not_rescued_mask):
        break  # All rescued!
        
    X_test_inc = X_failing_base.copy()
    # Apply increment and cap at MAX_HOURS_STUDIED
    new_hours = np.minimum(base_hours * (1 + inc), MAX_HOURS_STUDIED)
    X_test_inc['Hours_Studied'] = new_hours
    
    # Predict for all failing students
    preds_inc = model.predict(X_test_inc)
    
    # Check who crossed the threshold
    passed_mask = preds_inc >= PASS_THRESHOLD
    
    # Update those who passed AT THIS increment (and weren't passed before)
    newly_passed = passed_mask & not_rescued_mask
    required_increments[newly_passed] = inc

rescued_mask = ~np.isnan(required_increments)
rescued_increments = required_increments[rescued_mask]
unrescuable_count = len(failing_indices) - len(rescued_increments)

# =====================================================================
# 3. AGGREGATING RESULTS & HETEROGENEITY ANALYSIS
# =====================================================================
print("\n--- Individualized Minimal Intervention Results ---")
print(f"Total Failing Students: {len(failing_indices)}")
print(f"Students Rescued within +100% limit: {len(rescued_increments)} ({(len(rescued_increments)/len(failing_indices))*100:.1f}%)")
print(f"Students NOT Rescued (Exceed limits): {unrescuable_count} ({(unrescuable_count/len(failing_indices))*100:.1f}%)")

if len(rescued_increments) > 0:
    mean_inc = np.mean(rescued_increments) * 100
    median_inc = np.median(rescued_increments) * 100
    min_inc = np.min(rescued_increments) * 100
    max_inc = np.max(rescued_increments) * 100
    print(f"\nAmong Rescued Students (Required Increase %):")
    print(f"  Mean:   +{mean_inc:.2f}%")
    print(f"  Median: +{median_inc:.2f}%")
    print(f"  Range:  +{min_inc:.1f}% to +{max_inc:.1f}%")

    # Cumulative Rescued
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    print("\nCumulative Rescue Rates (% of Failing Students):")
    for t in thresholds:
        rescued_at_t = np.sum(rescued_increments <= t)
        print(f"  At <= {int(t*100)}% Increase: {rescued_at_t} students ({(rescued_at_t/len(failing_indices))*100:.1f}%)")

# Heterogeneity Analysis (Low / Medium / High baseline study hours)
print("\n--- Heterogeneity Analysis (By Baseline Hours) ---")
# Define groups based on terciles of the failing students' baseline hours
if len(failing_indices) > 0:
    p33 = np.percentile(base_hours, 33)
    p67 = np.percentile(base_hours, 67)
    
    group_low = base_hours <= p33
    group_med = (base_hours > p33) & (base_hours <= p67)
    group_high = base_hours > p67
    
    def analyze_group(name, mask):
        total = np.sum(mask)
        req_inc = required_increments[mask]
        rescued = req_inc[~np.isnan(req_inc)]
        if total > 0:
            pct_rescued = (len(rescued) / total) * 100
            med = np.median(rescued) * 100 if len(rescued) > 0 else np.nan
            print(f"  {name} Baseline (<= {p33:.1f} if Low, etc.): Total={total}, Rescued={len(rescued)} ({pct_rescued:.1f}%), Median Req. Increase=+{med:.1f}%")

    analyze_group("Low", group_low)
    analyze_group("Medium", group_med)
    analyze_group("High", group_high)

# =====================================================================
# 4. EXPORTING REPORTS AND PLOTS
# =====================================================================
output_dir = r"d:\ML\task\Scenario_Reports_and_Plots"
os.makedirs(output_dir, exist_ok=True)

# Export individual data to CSV
df_failing = X_failing_base.copy()
df_failing['Baseline_Predicted_Score'] = y_pred_all[failing_indices]
df_failing['Required_Increase_Pct'] = required_increments
df_failing['Rescued'] = ~np.isnan(required_increments)
df_failing.to_csv(os.path.join(output_dir, "Individualized_Minimal_Intervention.csv"), index=False)

# Plotting
sns.set_theme(style="whitegrid", context="paper")

if len(rescued_increments) > 0:
    # Plot 1: Distribution of Required Increases
    plt.figure(figsize=(8, 5))
    sns.histplot(rescued_increments * 100, bins=20, kde=True, color="purple")
    plt.title("Distribution of Minimum Required Study Hours Increase\n(For Rescued Students)", fontsize=14, weight='bold')
    plt.xlabel("Required Increase in Hours Studied (%)", fontsize=12)
    plt.ylabel("Number of Students", fontsize=12)
    plt.axvline(median_inc, color='r', linestyle='--', label=f'Median (+{median_inc:.1f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Fig5_Required_Increase_Distribution.png"), dpi=300)
    plt.close()

    # Plot 2: Cumulative Rescue Curve
    plt.figure(figsize=(8, 5))
    sorted_incs = np.sort(rescued_increments * 100)
    y_vals = np.arange(1, len(sorted_incs) + 1) / len(failing_indices) * 100
    
    # Add a 0 point
    sorted_incs = np.insert(sorted_incs, 0, 0)
    y_vals = np.insert(y_vals, 0, 0)
    
    plt.plot(sorted_incs, y_vals, drawstyle='steps-post', color="darkorange", linewidth=2.5)
    plt.title("Cumulative Rescue Curve: % of Failing Students Rescued vs. Effort", fontsize=14, weight='bold')
    plt.xlabel("Intervention Intensity (+% Hours Studied)", fontsize=12)
    plt.ylabel("% of Initially Failing Students Rescued", fontsize=12)
    plt.xlim(0, max(sorted_incs) + 5)
    plt.ylim(0, max(y_vals) + 5)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight the 35% mark to connect with previous analysis
    plt.axvline(35, color='gray', linestyle=':', label="Previous +35% Benchmark")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Fig6_Cumulative_Rescue_Curve.png"), dpi=300)
    plt.close()

print(f"\n✅ All analysis completed. Data and plots saved to '{output_dir}'.")
