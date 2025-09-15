import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


def create_explainer(feature_names, X_train):
    """Create LIME explainer for regression."""
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode="regression",
    )


def compute_lime_explanation(model, explainer, sample):
    """Compute LIME explanation for one sample."""
    exp = explainer.explain_instance(
        sample.flatten(), model.predict, num_features=len(explainer.feature_names)
    )
    return dict(exp.as_list())


def calculate_sensitivity(original_weights, perturbed_weights):
    """Calculate absolute difference between original and perturbed LIME explanations."""
    return {
        feature: abs(
            original_weights.get(feature, 0) - perturbed_weights.get(feature, 0)
        )
        for feature in original_weights.keys()
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
    Perform LIME sensitivity analysis on a regression dataset.

    Parameters:
        model: Trained regression model.
        X_train: Training features (DataFrame).
        y_train: Training target (Series).
        X_test: Test features (DataFrame).
        y_test: Test target (Series).
        sample_index: Index of the test sample to explain.
        epsilon: Std deviation of noise for perturbation.
        verbose: Print detailed output.

    Returns:
        sensitivity (dict): Feature sensitivity values.
    """

    feature_names = X_train.columns.tolist()
    explainer = create_explainer(feature_names, X_train)

    sample = X_test.iloc[sample_index].values.reshape(1, -1)
    original_weights = compute_lime_explanation(model, explainer, sample)

    perturbed_sample = sample + np.random.normal(0, epsilon, sample.shape)
    perturbed_weights = compute_lime_explanation(model, explainer, perturbed_sample)

    sensitivity = calculate_sensitivity(original_weights, perturbed_weights)

    if verbose:
        print("📊 LIME Sensitivity for each feature:")
        for feature, diff in sensitivity.items():
            print(f"{feature}: {diff:.4f}")

    return sensitivity
