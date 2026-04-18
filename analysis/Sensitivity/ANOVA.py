import pandas as pd
import statsmodels.api as sm

# -------------------- 1. Load data --------------------
DATA_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="Z-Score")
target_column = df.columns[-1]

# -------------------- 2. Separate features and target --------------------
X = df.drop(columns=[target_column])
y = df[target_column]

# -------------------- 3. Calculate F-statistic and p-value for each feature --------------------
results = []
for feature in X.columns:
    X_feature = sm.add_constant(X[feature])  # add intercept
    model = sm.OLS(y, X_feature).fit()
    f_stat = model.fvalue           # F-statistic
    p_value = model.f_pvalue        # p-value
    results.append({
        "Feature": feature,
        "F-statistic": f_stat,
        "p-value": p_value
    })

anova_like_df = pd.DataFrame(results)

# -------------------- 4. Display and copy --------------------
print(anova_like_df)
anova_like_df.to_clipboard(index=False)
print("✅ F-statistic and p-values copied to clipboard")
