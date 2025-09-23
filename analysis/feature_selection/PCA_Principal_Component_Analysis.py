import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load data
sheet_name = "LabelEncoded"
data = pd.read_excel(
    r"D:\ML\Main_utils\Task\GLEMETA_MADDPG_Final_IoT_MEC_UAV_Dataset.xlsx",
    sheet_name=sheet_name,
)
X = data.drop("offload_ratio", axis=1)
y = data["offload_ratio"]

# Step 2: Preprocess data (handle missing values, if any)
if X.isnull().sum().sum() > 0:
    print("Missing values detected. Filling with mean...")
    X = X.fillna(X.mean())

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA to get feature importance
pca = PCA()
pca.fit(X_scaled)

# Step 5: Calculate feature importance based on absolute loadings
# Sum absolute loadings across components that explain 95% of variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to explain 95% of variance: {n_components_95}")

# Get absolute loadings for the first n_components
loadings = np.abs(pca.components_[:n_components_95])
feature_importance = np.sum(loadings, axis=0)
feature_importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": feature_importance}
)
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=True
)

# Step 6: Identify least important features (e.g., bottom 10%)
n_features = X.shape[1]
n_remove = max(3, int(0.1 * n_features))  # Remove bottom 10% of features
least_important_features = feature_importance_df.head(n_remove)["Feature"].tolist()
print(f"Least important features to remove: {least_important_features}")

# Step 7: Remove least important features from original dataset
X_reduced = X.drop(columns=least_important_features)
print(f"Original dataset shape: {X.shape}")
print(f"Reduced dataset shape: {X_reduced.shape}")

# Step 8: Save the reduced dataset
reduced_data = X_reduced.copy()
reduced_data["Market Share of AI Companies (%)"] = y
