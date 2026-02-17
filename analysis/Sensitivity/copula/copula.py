from couples_sensitivity_analysis import couples_sensitivity_analysis
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
dt = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    sheet_name="Data_after_KFold_DTC",
)

target_column = dt.columns[-1]
X = dt.drop(target_column, axis=1)
y = dt[target_column]

features = X.columns

# Train the Decision Tree Classifier
model = DecisionTreeClassifier(
        max_depth=7,  # If accuracy is still 1.0, lower this to 2 or 3
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
copula = couples_sensitivity_analysis(
    model,
    X,
    y,
    feature_pairs,
    "accuracy",   # changed from "mse"
    40
)

# Display the sensitivity report
print(copula)
copula.to_clipboard()
