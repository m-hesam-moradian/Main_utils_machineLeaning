import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from matplotlib.colors import LinearSegmentedColormap

# =====================================================================
# 1. SETUP & DATA PREPARATION
# =====================================================================
excel_path = r"d:\ML\task\Data.xlsx"
sheet_name = "Optimal_Data_MRMR"

print("Loading data for advanced plotting...")
try:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
except FileNotFoundError:
    print(f"Error: Could not find {excel_path}")
    exit()

target_column = df.columns[-1]
X = df.drop(columns=target_column)
y = df[target_column]

split_idx = int(len(df) * 0.8)
X_train = X[:split_idx]
y_train = y[:split_idx]

model = XGBRegressor(n_estimators=1000, max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred_all = model.predict(X)

output_dir = r"d:\ML\task\Scenario_Reports_and_Plots\Advanced_Figures"
os.makedirs(output_dir, exist_ok=True)
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}) # Transparent for ridge plot

PASS_THRESHOLD = 60.0
MAX_HOURS = 37.0
MIN_RES = 0.0

failing_indices = np.where(y_pred_all < PASS_THRESHOLD)[0]
passing_indices = np.where(y_pred_all >= PASS_THRESHOLD)[0]

# =====================================================================
# FIGURE 1: RIDGE PLOT (JOYPLOT) OF INTERVENTION EVOLUTION
# =====================================================================
print("Generating Figure 1: Ridge Plot (Intervention Evolution)...")
# We will simulate 0%, 10%, 20%, 30%, 40%, 50% increase
steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
ridge_data = []

for step in steps:
    X_sim = X.copy()
    X_sim['Hours_Studied'] = np.minimum(X_sim['Hours_Studied'] * (1 + step), MAX_HOURS)
    preds = model.predict(X_sim)
    for p in preds:
        ridge_data.append({'Intervention': f"+{int(step*100)}% Effort", 'Predicted_Score': p})

df_ridge = pd.DataFrame(ridge_data)

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(len(steps), rot=-.25, light=.7)
g = sns.FacetGrid(df_ridge, row="Intervention", hue="Intervention", aspect=8, height=1.2, palette=pal)

# Draw the densities
g.map(sns.kdeplot, "Predicted_Score", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "Predicted_Score", clip_on=False, color="w", lw=2, bw_adjust=.5)

# Passing threshold line
g.map(plt.axvline, x=PASS_THRESHOLD, color='red', linestyle='--', linewidth=2, zorder=10)

# Add label to each plot
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes, fontsize=12)
g.map(label, "Predicted_Score")

g.figure.subplots_adjust(hspace=-0.5) # Overlap plots
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.figure.suptitle("Evolution of Cohort Performance Distribution Across Intervention Intensities", 
                  fontsize=16, weight='bold', y=1.05)
plt.savefig(os.path.join(output_dir, "FigA_RidgePlot_Evolution.svg"), format='svg', bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 2: 2D BIVARIATE KDE CONTOUR (HETEROGENEITY MAP)
# =====================================================================
print("Generating Figure 2: Bivariate KDE Contour...")
# Recalculate Minimal Intervention for failing students
X_failing = X.iloc[failing_indices].copy()
base_hours = X_failing['Hours_Studied'].values
req_inc = np.full(len(failing_indices), np.nan)

incs = np.arange(0.01, 1.01, 0.01)
for inc in incs:
    mask = np.isnan(req_inc)
    if not np.any(mask): break
    X_tmp = X_failing.copy()
    X_tmp['Hours_Studied'] = np.minimum(base_hours * (1 + inc), MAX_HOURS)
    p = model.predict(X_tmp)
    req_inc[(p >= PASS_THRESHOLD) & mask] = inc

rescued = ~np.isnan(req_inc)
df_contour = pd.DataFrame({
    'Baseline_Hours': base_hours[rescued],
    'Required_Increase_Pct': req_inc[rescued] * 100
})

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(8, 7))

# Draw a 2D KDE plot with contour lines
sns.kdeplot(
    data=df_contour, x="Baseline_Hours", y="Required_Increase_Pct",
    fill=True, cmap="mako", levels=15, thresh=.05, ax=ax, alpha=0.8
)
# Overlay the scatter
sns.scatterplot(
    data=df_contour, x="Baseline_Hours", y="Required_Increase_Pct",
    color="white", s=20, edgecolor="black", linewidth=0.5, alpha=0.7, ax=ax
)

ax.set_title("Intervention Landscape: Baseline Effort vs. Required Nudge", fontsize=15, weight='bold', pad=15)
ax.set_xlabel("Baseline Hours Studied (Failing Students)", fontsize=13, weight='bold')
ax.set_ylabel("Minimal Required Intervention (+ %)", fontsize=13, weight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig(os.path.join(output_dir, "FigB_Bivariate_KDE_Heterogeneity.svg"), format='svg', bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 3: RAINCLOUD-STYLE PLOT (VIOLIN + STRIP)
# =====================================================================
print("Generating Figure 3: Raincloud/Violin Plot...")
# Group failing students into Low, Med, High
p33, p67 = np.percentile(base_hours, 33), np.percentile(base_hours, 67)
groups = []
for h in df_contour['Baseline_Hours']:
    if h <= p33: groups.append('Low Baseline')
    elif h <= p67: groups.append('Medium Baseline')
    else: groups.append('High Baseline')
df_contour['Baseline_Group'] = groups

# Order groups
order = ['Low Baseline', 'Medium Baseline', 'High Baseline']

f, ax = plt.subplots(figsize=(10, 6))
# Violin plot (the "cloud")
sns.violinplot(
    data=df_contour, x="Baseline_Group", y="Required_Increase_Pct",
    order=order, inner="quartile", palette="muted", split=True, alpha=0.5, ax=ax, linewidth=2
)
# Strip plot (the "rain")
sns.stripplot(
    data=df_contour, x="Baseline_Group", y="Required_Increase_Pct",
    order=order, color="black", alpha=0.4, jitter=0.15, size=5, ax=ax
)

ax.set_title("Heterogeneous Impact: Required Intervention Across Baseline Profiles", fontsize=15, weight='bold', pad=15)
ax.set_xlabel("Student Baseline Profile", fontsize=13, weight='bold')
ax.set_ylabel("Minimal Required Increase (%)", fontsize=13, weight='bold')
plt.savefig(os.path.join(output_dir, "FigC_Raincloud_Heterogeneity.svg"), format='svg', bbox_inches='tight')
plt.close()


# =====================================================================
# FIGURE 4: CAUSAL IMPACT TRAJECTORY (NEGATIVE SCENARIO)
# =====================================================================
print("Generating Figure 4: Causal Impact Trajectory...")
# Recalculate Negative Sensitivity
steps_neg = np.arange(0.00, 0.55, 0.05)
neg_means = []
base_mean_neg = np.mean(y_pred_all)

for step in steps_neg:
    X_sim = X.copy()
    if 'Tutoring_Sessions' in X_sim.columns: X_sim['Tutoring_Sessions'] = np.maximum(X_sim['Tutoring_Sessions'] * (1 - step), MIN_RES)
    if 'Internet_Access' in X_sim.columns: X_sim['Internet_Access'] = np.maximum(X_sim['Internet_Access'] * (1 - step), MIN_RES)
    preds = model.predict(X_sim)
    neg_means.append(preds)

neg_means = np.array(neg_means) # Shape: (steps, students)
mean_trajectory = np.mean(neg_means, axis=1)
# Calculate confidence intervals (95%) using std dev
std_trajectory = np.std(neg_means, axis=1) / np.sqrt(len(y_pred_all))
ci_upper = mean_trajectory + 1.96 * std_trajectory
ci_lower = mean_trajectory - 1.96 * std_trajectory

plt.figure(figsize=(9, 6))
plt.plot(steps_neg * 100, mean_trajectory, color="firebrick", linewidth=3, marker='D', markersize=8, label="Mean Predicted Cohort Score")
plt.fill_between(steps_neg * 100, ci_lower, ci_upper, color="firebrick", alpha=0.2, label="95% Confidence Interval")

plt.title("Trajectory of Systemic Vulnerability: Resource Deprivation", fontsize=15, weight='bold')
plt.xlabel("Resource Reduction (Tutoring & Internet %)", fontsize=13, weight='bold')
plt.ylabel("Expected Average Score", fontsize=13, weight='bold')
plt.axhline(PASS_THRESHOLD, color='black', linestyle='--', linewidth=2, label="Passing Threshold")
plt.legend(loc="upper right", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig(os.path.join(output_dir, "FigD_Trajectory_Vulnerability.svg"), format='svg', bbox_inches='tight')
plt.close()

print(f"\n[SUCCESS] Advanced Academic Figures generated in '{output_dir}'.")
