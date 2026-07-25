# Make sure you have this library installed in your environment
from couples_sensitivity_analysis import couples_sensitivity_analysis

import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Changed from Regressor to Classifier
from sklearn.metrics import accuracy_score # Import a classification metric

# Load the dataset
# Ensure the target column in this sheet contains your class labels
dt = pd.read_excel(
    r"C:\\Users\\Sam\\Desktop\\ML\\task\\Data.xlsx", sheet_name="Data_after_KFold_ETC(RFE)"
)

target_column = dt.columns[-1]
X = dt.drop(target_column, axis=1)
y = dt[target_column]
features = X.columns

# Train the RandomForestClassifier model
# Using RandomForestClassifier for classification tasks
# model = DecisionTreeClassifier(
#     max_depth=10,
#     min_samples_split=2,
#     min_samples_leaf=1

# )
from sklearn.ensemble import ExtraTreesClassifier   # <-- changed import

model = ExtraTreesClassifier(
    
        # max_depth=3,
        # n_estimators=5,
        # max_depth=6,
        # n_estimators=20,
        max_depth=7,
        n_estimators=10,

        random_state=42
)

model.fit(X, y)

# Define pairs of features for sensitivity analysis
feature_pairs = [
    (features[i], features[j])
    for i in range(len(features))
    for j in range(len(features))
]

# Perform the couples sensitivity analysis
# Changed the metric from "mse" to "accuracy" for classification
copula = couples_sensitivity_analysis(model, X, y, feature_pairs, "accuracy", 40)

# Display the sensitivity report
print(copula)

# to_clipboard might not work for a complex object, but keeping it as in your original code
try:
    copula.to_clipboard()
    print("\nSuccessfully copied to clipboard!")
except Exception as e:
    print(f"\nCould not copy to clipboard: {e}")

