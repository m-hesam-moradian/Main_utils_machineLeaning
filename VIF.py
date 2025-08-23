import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = 'Vehicle-Specific and Traffic _dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='sim_11')
df=df.dropna()
X = df.drop("Vehicle Speed", axis=1)

y=df["Vehicle Speed"]
df = df.dropna()

# Set initial features (excluding target)
features = list(df.drop("Vehicle Speed", axis=1).columns)

all_vif_data = []

while True:
    print("✅ Current features:", features)

    # Prepare the feature matrix with one-hot encoding
    X = pd.get_dummies(df[features], drop_first=True)

    # Remove non-numeric columns if any
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print("⛔️ Non-numeric columns detected and excluded:", non_numeric)
        X = X.drop(columns=non_numeric)

    # Add constant term for intercept
    X_with_const = np.column_stack((np.ones(X.shape[0]), X.values))
    columns_with_const = ['const'] + list(X.columns)

    # Create DataFrame with constant
    X_df = pd.DataFrame(X_with_const, columns=columns_with_const)

    # Calculate VIFs
    try:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_df.columns
        vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

        # Save VIFs for this iteration
        all_vif_data.append(vif_data)

        # Drop constant for max VIF check
        vif_no_const = vif_data[vif_data["Variable"] != "const"]
        max_vif = vif_no_const["VIF"].max()
        highest_vif_var = vif_no_const.loc[vif_no_const["VIF"].idxmax(), "Variable"]

        print(vif_data)
        print("🚨 Highest VIF:", highest_vif_var, "=", max_vif)

        # Stop if VIF is acceptable (e.g. below 5 or 10)
        if max_vif < 1:
            print("✅ All VIFs are below threshold. Stopping.")
            break

        # Remove the variable with the highest VIF
        if highest_vif_var in features:
            features.remove(highest_vif_var)
        else:
            # In case of one-hot encoded column, find corresponding original column
            for f in features:
                if highest_vif_var.startswith(f + '_') or highest_vif_var == f:
                    features.remove(f)
                    break

    except Exception as e:
        print(f"❌ Error calculating VIF: {e}")
        break

# Combine all VIF data if needed
vd = pd.concat(all_vif_data, ignore_index=True)

# Save to Excel
vd.to_excel("vif_results.xlsx", index=False)
