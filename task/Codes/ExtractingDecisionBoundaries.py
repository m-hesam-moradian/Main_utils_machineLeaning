import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load Excel data
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="Data_after_KFold")
target_column = "Anomalous Load"
y = LabelEncoder().fit_transform(df[target_column])
X = df.drop(columns=[target_column])

# Optional: reduce to 2D for visualization if X has more than 2 features
if X.shape[1] > 2:
    X = PCA(n_components=2).fit_transform(X)

# Train classifier
clf = LogisticRegression()
clf.fit(X, y)

# Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict over grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()