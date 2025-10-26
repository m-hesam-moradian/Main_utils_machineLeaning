import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# Load Excel file and specify the sheet name
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="BSS.No.1-Target 1")  # Replace with actual file and sheet name

# Separate features and target
target_col = 'Anomalous Load'
X = df.drop(columns=[target_col])
y = df[target_col]

# Optional: scale features for VIF stability
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Calculate VIF and filter features
def calculate_vif(df, threshold=5.0):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data[vif_data["VIF"] < threshold]

# Apply VIF filtering
vif_result = calculate_vif(X_scaled)
selected_features = vif_result["feature"].tolist()

# Final selected features DataFrame
X_selected = X[selected_features]

# Optional: Combine with target if needed
selected_df = pd.concat([X_selected, y], axis=1)

# Output
print("Selected features based on VIF < 5:")
print(vif_result)
vif_result.to_clipboard(index=False)