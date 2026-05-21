import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lime.lime_tabular import LimeTabularExplainer

# ---------------------------------------------------------
# LIME FUNCTIONS FOR REGRESSION
# ---------------------------------------------------------
def create_explainer(feature_names, X_train):
    """Create LIME explainer for Regression."""
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=None,
        mode="regression"
    )

def compute_lime_explanation(model, explainer, sample):
    """Compute LIME explanation for one sample."""
    exp = explainer.explain_instance(
        sample.flatten(),
        model.predict,
        num_features=len(explainer.feature_names)
    )
    return dict(exp.as_list())

def calculate_sensitivity(original_weights, perturbed_weights):
    """Calculate absolute difference between LIME explanations."""
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
    X_test,
    sample_index=5,
    epsilon=0.05,
    verbose=True,
):
    feature_names = X_train.columns.tolist()
    explainer = create_explainer(feature_names, X_train)

    sample = X_test.iloc[sample_index].values.reshape(1, -1)

    # Original explanation
    original_weights = compute_lime_explanation(model, explainer, sample)

    # Perturbed sample
    perturbed_sample = sample + np.random.normal(0, epsilon, sample.shape)

    # Perturbed explanation
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
dt = pd.read_excel(
    r'C:\Users\Sam\Desktop\ML\task\Data.xlsx',
    sheet_name='Data_after_KFold_XGBoost'
)

target_column = dt.columns[-1]

X = dt.drop(columns=[target_column])
y = dt[target_column].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------------
# MODELS (ENR & XGBR)
# ---------------------------------------------------------

# 1. ElasticNet Regressor (ENR)
enr_model = ElasticNet(
    alpha=1.0, 
    l1_ratio=0.5, 
    random_state=42
)
enr_model.fit(X_train, y_train)

# 2. XGBoost Regressor (XGBR)
xgbr_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
xgbr_model.fit(X_train, y_train)

# ---------------------------------------------------------
# REGRESSION METRICS
# ---------------------------------------------------------
def regression_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"{name}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print("-" * 30)

# Predictions
enr_pred = enr_model.predict(X_test)
xgbr_pred = xgbr_model.predict(X_test)

# Metrics
regression_metrics("ElasticNet (ENR)", y_test, enr_pred)
regression_metrics("XGBRegressor (XGBR)", y_test, xgbr_pred)

print("\n" + "=" * 30 + "\n")

# ---------------------------------------------------------
# LIME SENSITIVITY ANALYSIS (BOTH MODELS)
# ---------------------------------------------------------

lime_ENR = lime_sensitivity_analysis(
    model=enr_model,
    X_train=X_train,
    X_test=X_test,
    sample_index=5,
    epsilon=0.05
)

lime_XGBR = lime_sensitivity_analysis(
    model=xgbr_model,
    X_train=X_train,
    X_test=X_test,
    sample_index=5,
    epsilon=0.05
)

# Convert to DataFrames
lime_ENR_df = pd.DataFrame.from_dict(
    lime_ENR,
    orient='index',
    columns=['ENR Sensitivity']
)

lime_XGBR_df = pd.DataFrame.from_dict(
    lime_XGBR,
    orient='index',
    columns=['XGBR Sensitivity']
)

# Merge for comparison report
final_report = pd.concat([lime_ENR_df, lime_XGBR_df], axis=1)

print("\n📊 FINAL LIME SENSITIVITY COMPARISON")
print(final_report)

# Export
final_report.to_clipboard(index=True, header=True)
print("\n✅ Table copied to clipboard!")