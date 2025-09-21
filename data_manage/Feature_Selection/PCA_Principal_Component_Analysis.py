import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load data
sheet_name = "LabelEncoded"
data = pd.read_excel(
    r"D:\ML\Main_utils\Task\Global_AI_Content_Impact_Dataset.xlsx",
    sheet_name=sheet_name,
)
X = data.drop("Market Share of AI Companies (%)", axis=1)
y = data["Market Share of AI Companies (%)"]

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Drop rows with NaN in target
if y.isnull().any():
    mask = ~y.isnull()
    X_imputed = X_imputed[mask]
    y = y[mask].reset_index(drop=True)

# Rename columns
X = pd.DataFrame(X_imputed, columns=[f"V {i+1}" for i in range(X_imputed.shape[1])])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

pca = PCA(n_components=10)  # or whatever number you choose
X_pca = pca.fit_transform(X)

loadings = pd.DataFrame(pca.components_.T, index=X.columns)


top_features_per_pc = {}
for i in range(pca.n_components):
    pc_loadings = loadings.iloc[:, i].abs().sort_values(ascending=False)
    top_features_per_pc[f"PC{i+1}"] = pc_loadings.index.tolist()


plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title(f"PCA Projection for {sheet_name}")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.show()

# KMeans clustering on PCA-reduced data
k_values = [2, 3, 4, 5]
fig, axes = plt.subplots(1, len(k_values), figsize=(15, 4))
for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X_pca)
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", edgecolor="k")
    axes[i].set_title(f"KMeans (k={k})")
plt.show()

# RandomForest on PCA-selected features
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_pca, y)
importances = rf.feature_importances_

# Visualize PCA component importances
plt.figure(figsize=(8, 6))
plt.bar(
    range(n_components),
    importances,
    tick_label=[f"PC{i+1}" for i in range(n_components)],
)
plt.title("Feature Importance via PCA Components")
plt.xlabel("Principal Component")
plt.ylabel("Importance")
plt.show()
