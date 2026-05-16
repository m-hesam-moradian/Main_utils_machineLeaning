import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import scipy.special as special  # <-- Added for extreme mathematical limits
import numpy as np

# -------------------- 1. Configuration & Load Data --------------------
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
INPUT_SHEET = "DATA_Shuffled"     
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
    p_value_numeric = model.f_pvalue        
    
    status = "Kept" if p_value_numeric < P_VALUE_THRESHOLD else "Removed"
    
    # ---> ADVANCED MATH TO BYPASS THE 0.0 & INFINITY LIMITS <---
    if np.isinf(f_stat):
        # If F-statistic is literally infinity (perfect 1:1 correlation)
        display_p_value = "0.0000E+00"
    elif p_value_numeric == 0.0:
        # Calculate the natural log of the p-value
        log_p = stats.f.logsf(f_stat, model.df_model, model.df_resid)
        
        # If the number is SO small that even the log survival function underflows to -infinity
        if np.isinf(log_p):
            # Fallback: Asymptotic approximation using incomplete beta function limits
            d1 = model.df_model
            d2 = model.df_resid
            x = d2 / (d2 + d1 * f_stat)
            a = d2 / 2.0
            b = d1 / 2.0
            
            # Mathematical formula to approximate the logarithm of the p-value safely
            log_p = a * np.log(x) + b * np.log(1 - x) - np.log(a) - special.betaln(a, b)

        # Convert natural log to Base 10
        log10_p = log_p / np.log(10)
        # Separate the exponent and the mantissa to build our own scientific string
        exponent = int(np.floor(log10_p))
        mantissa = 10**(log10_p - exponent)
        display_p_value = f"{mantissa:.4f}E{exponent}"
    else:
        # Format regular p-values into standard scientific notation strings
        display_p_value = f"{p_value_numeric:.4E}"
    # ----------------------------------------------------
        
    anova_results.append({
        "Feature": feature,
        "F-statistic": f_stat,
        "p-value": display_p_value,  # Writing the new string notation to the report
        "Status": status
    })

# Convert to DataFrame and sort by F-statistic descending (highest impact at the top)
anova_df = pd.DataFrame(anova_results).sort_values(by="F-statistic", ascending=False)

# -------------------- 3. Prepare the New Dataset --------------------
kept_features = anova_df[anova_df["Status"] == "Kept"]["Feature"].tolist()
removed_features = anova_df[anova_df["Status"] == "Removed"]["Feature"].tolist()

print(f"Features kept: {len(kept_features)}")
print(f"Features removed: {len(removed_features)}")

new_dataset = df[kept_features + [target_column]]

# -------------------- 4. Save back to the SAME Excel File --------------------
print("\nSaving report and new dataset back to the same Excel file...")

with pd.ExcelWriter(DATA_PATH, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    anova_df.to_excel(writer, sheet_name="ANOVA_Report", index=False)
    new_dataset.to_excel(writer, sheet_name="Data_After_ANOVA", index=False)

print(f"✅ Success! Data appended to {DATA_PATH} in new sheets: 'ANOVA_Report' and 'Data_After_ANOVA'.")

if removed_features:
    print(f"List of removed features: {removed_features}")