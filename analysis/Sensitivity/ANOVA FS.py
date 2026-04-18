import pandas as pd
import statsmodels.api as sm

# -------------------- 1. Configuration & Load Data --------------------
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
INPUT_SHEET = "Z-Score"     
P_VALUE_THRESHOLD = 0.05  # Features with p-value >= 0.05 will be removed

print(f"Loading data from {DATA_PATH}...")
df = pd.read_excel(DATA_PATH, sheet_name=INPUT_SHEET)

# Assuming the target is the last column
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# -------------------- 2. Calculate ANOVA (F-value & p-value) --------------------
print("Running ANOVA feature selection...")
anova_results = []

for feature in X.columns:
    X_feature = sm.add_constant(X[feature])  # Add intercept
    model = sm.OLS(y, X_feature).fit()
    f_stat = model.fvalue           
    p_value = model.f_pvalue        
    
    # Based on the article: keep features with substantial predictive power.
    # We use p < 0.05 as the statistical cutoff for significance.
    status = "Kept" if p_value < P_VALUE_THRESHOLD else "Removed"
        
    anova_results.append({
        "Feature": feature,
        "F-statistic": f_stat,
        "p-value": p_value,
        "Status": status
    })

# Convert to DataFrame and sort by F-statistic descending (highest impact at the top)
anova_df = pd.DataFrame(anova_results).sort_values(by="F-statistic", ascending=False)

# -------------------- 3. Prepare the New Dataset --------------------
# Filter only the features that were "Kept"
kept_features = anova_df[anova_df["Status"] == "Kept"]["Feature"].tolist()
removed_features = anova_df[anova_df["Status"] == "Removed"]["Feature"].tolist()

print(f"Features kept: {len(kept_features)}")
print(f"Features removed: {len(removed_features)}")

# Create the new dataset with only the selected features + target
new_dataset = df[kept_features + [target_column]]

# -------------------- 4. Save back to the SAME Excel File --------------------
print("\nSaving report and new dataset back to the same Excel file...")

# Using 'mode="a"' (append) and engine='openpyxl' opens the existing file, 
# adds the new sheets, and closes it without overwriting the original data.
with pd.ExcelWriter(DATA_PATH, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    
    # Sheet 1: The detailed report showing F-values, p-values, and what was removed
    anova_df.to_excel(writer, sheet_name="ANOVA_Report", index=False)
    
    # Sheet 2: The clean dataset ready for your ML models
    new_dataset.to_excel(writer, sheet_name="Data_After_ANOVA", index=False)

print(f"✅ Success! Data appended to {DATA_PATH} in new sheets: 'ANOVA_Report' and 'Data_After_ANOVA'.")

if removed_features:
    print(f"List of removed features: {removed_features}")