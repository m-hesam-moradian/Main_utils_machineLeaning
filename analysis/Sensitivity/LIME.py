import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer

# ---------------------------------------------------------
# UPDATED LIME FUNCTIONS FOR CLASSIFICATION
# ---------------------------------------------------------
def create_explainer(feature_names, X_train):
    """Create LIME explainer for Classification."""
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['0', '1', '2', '3', '4', '5'], # Generic class names
        mode="classification"  # <--- CHANGED to classification
    )

def compute_lime_explanation(model, explainer, sample):
    """Compute LIME explanation for one sample using predict_proba."""
    # For classification, we use predict_proba
    exp = explainer.explain_instance(
        sample.flatten(), 
        model.predict_proba, # <--- CHANGED to predict_proba
        num_features=len(explainer.feature_names)
    )
    return dict(exp.as_list())

def calculate_sensitivity(original_weights, perturbed_weights):
    """Calculate absolute difference between original and perturbed LIME explanations."""
    # We combine keys from both to ensure we don't miss features if LIME returns different top features
    all_features = set(original_weights.keys()) | set(perturbed_weights.keys())
    
    return {
        feature: abs(
            original_weights.get(feature, 0) - perturbed_weights.get(feature, 0)
        )
        for feature in all_features
    }

def lime_sensitivity_analysis(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    sample_index=5,
    epsilon=0.05,
    verbose=True,
):
    """
    Perform LIME sensitivity analysis.
    """
    feature_names = X_train.columns.tolist()
    explainer = create_explainer(feature_names, X_train)

    # Grab the sample
    sample = X_test.iloc[sample_index].values.reshape(1, -1)
    
    # Get original explanation
    original_weights = compute_lime_explanation(model, explainer, sample)

    # Perturb the sample (add noise)
    perturbed_sample = sample + np.random.normal(0, epsilon, sample.shape)
    
    # Get perturbed explanation
    perturbed_weights = compute_lime_explanation(model, explainer, perturbed_sample)

    sensitivity = calculate_sensitivity(original_weights, perturbed_weights)

    if verbose:
        print(f"📊 LIME Sensitivity for {type(model).__name__}:")
        for feature, diff in sensitivity.items():
            print(f"{feature}: {diff:.4f}")
        print("-" * 30)

    return sensitivity

# ---------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------

# Load data
# Using raw string (r'') to avoid path errors on Windows
dt = pd.read_excel(r'C:\Users\Sam\Desktop\ML\task\Data.xlsx', sheet_name='Data_after_KFold_ADAC')

# Identify target column (assuming last column)
target_column = dt.columns[-1]

# Prepare the features and target
X = dt.drop(columns=[target_column])
y = dt[target_column]

# Ensure y is integer for classification
y = y.astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. K-Nearest Neighbors Classification (KNNC)
knnc_model = KNeighborsClassifier(n_neighbors=5)
knnc_model.fit(X_train, y_train)

# 2. Adaptive Gradient Boosting Classification (ADAC)
adac_model = AdaBoostClassifier(n_estimators=100, random_state=42)
adac_model.fit(X_train, y_train)

# Evaluate Models
print("KNNC Accuracy:", accuracy_score(y_test, knnc_model.predict(X_test)))
print("ADAC Accuracy:", accuracy_score(y_test, adac_model.predict(X_test)))
print("\n" + "="*30 + "\n")

# ---------------------------------------------------------
# Perform LIME sensitivity analysis
# ---------------------------------------------------------

# CORRECTED FUNCTION CALLS:
# Must pass: model, X_train, y_train, X_test, y_test
# lime_KNNC = lime_sensitivity_analysis(
#     model=knnc_model, 
#     X_train=X_train, 
#     y_train=y_train, 
#     X_test=X_test, 
#     y_test=y_test, 
#     sample_index=5, 
#     epsilon=0.05
# )

lime_ADAC = lime_sensitivity_analysis(
    model=adac_model, 
    X_train=X_train, 
    y_train=y_train, 
    X_test=X_test, 
    y_test=y_test, 
    sample_index=5, 
    epsilon=0.05
)
# lime_KNNC_df = pd.DataFrame.from_dict(lime_KNNC, orient='index', columns=['KNNC Sensitivity'])
# lime_KNNC_df.to_clipboard(index=True, header=True)
lime_ADAC_df = pd.DataFrame.from_dict(lime_ADAC, orient='index', columns=['ADAC Sensitivity'])
lime_ADAC_df.to_clipboard(index=True, header=True)

