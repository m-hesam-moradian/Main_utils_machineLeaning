# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:57:49 2024

@author: sinap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  # Added for handling NaN

sheet_name = "LabelEncoded"
# Load the data
data = pd.read_excel(
    "D:\ML\Main_utils\Task\Global_AI_Content_Impact_Dataset.xlsx", sheet_name=sheet_name
)
X = data.drop("Market Share of AI Companies (%)", axis=1)  # Features
y = data["Market Share of AI Companies (%)"]  # Target

# Handle missing values in X by imputing with the mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# If y has NaN values, drop those rows (since y is used in RandomForest later)
if y.isnull().any():
    mask = ~y.isnull()
    X_imputed = X_imputed[mask]
    y = y[mask].reset_index(drop=True)

# Convert X_imputed back to DataFrame for consistency with later code
params = [f"V {i+1}" for i in range(X_imputed.shape[1])]
X = pd.DataFrame(X_imputed, columns=params)

# Train-test split (though not used in KMeans or PCA in your code)
X_tr, X_te, Y_tr, Y_t = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# PCA for feature grouping
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting PCA results with arrows
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)


# Correlation function to find closely related features
def correlation(dataset, threshold):
    closely_related_indices = []
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                closely_related_indices.append((i, j))
    return col_corr, closely_related_indices


corr_features, closely_related_indices = correlation(X_train, 0.8)

# Arrow annotations for closely related features
arrow_length = 50.5
for i, j in closely_related_indices:
    pca_cop_i = pca.components_[:, i]
    pca_cop_j = pca.components_[:, j]
    arrow_i = arrow_length * pca_cop_i
    arrow_j = arrow_length * pca_cop_j

    plt.arrow(
        0,
        0,
        arrow_i[0],
        arrow_i[1],
        color="r",
        alpha=0.5,
        head_width=0.05,
        linewidth=2.0,
    )
    plt.arrow(
        0,
        0,
        arrow_j[0],
        arrow_j[1],
        color="b",
        alpha=0.5,
        head_width=0.05,
        linewidth=2.0,
    )

    text_offset = 0.1
    plt.text(
        arrow_i[0] + text_offset,
        arrow_i[1] + text_offset,
        str(i),
        color="g",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    plt.text(
        arrow_j[0] + text_offset,
        arrow_j[1] + text_offset,
        str(j),
        color="g",
        ha="right",
        va="bottom",
        fontsize=8,
    )

plt.title(f"PCA with Arrows for {sheet_name}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# KMeans clustering for different k values
k_values = [2, 3, 4, 5]
fig, axes = plt.subplots(1, len(k_values), figsize=(15, 4))

for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)  # Now X has no NaN values
    labels = kmeans.labels_
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", edgecolor="k")
    axes[i].set_title(f"KMeans (k={k})")
plt.show()

# Feature Importance using RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_

# Sort features by importance
sorted_indices = np.argsort(importances)[::-1]
top_features = sorted_indices[:45]

# Visualize feature importance
plt.figure(figsize=(8, 6))
plt.bar(range(len(top_features)), importances[top_features], tick_label=top_features)
plt.title("Feature Importance Ranking")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
