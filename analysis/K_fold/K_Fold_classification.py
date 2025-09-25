import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from copy import deepcopy

# --- Load dataset ---
sheet_name = "Data after K-FOLD (votingC)"
df = pd.read_excel(
    r"D:\ML\Main_utils\task\WA_Fn-UseC_-HR-Employee-Attrition.xlsx",
    sheet_name=sheet_name,
)
target_column = "Attrition"

# Convert target to categorical (example: binarize based on median)
# Adjust this logic based on your specific classification needs
median_value = df[target_column].median()
df[target_column] = (df[target_column] > median_value).astype(
    int
)  # Binary classification: 0 or 1

# Features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Define models ---
models = {
    "LDA": LinearDiscriminantAnalysis(
        solver="svd",  # Default: Singular Value Decomposition
        shrinkage=None,  # Default: No shrinkage for 'svd'
        priors=None,  # Default: Priors estimated from class frequencies
        n_components=None,  # Default: min(n_classes - 1, n_features)
        store_covariance=False,  # Default: Do not compute covariance matrix
        tol=1e-4,  # Default: Tolerance for singular values
        covariance_estimator=None,  # Default: Use standard covariance estimation
    ),
    "VotingC": VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,  # Default: 100 trees
                    criterion="gini",  # Default: Gini impurity for classification
                    max_depth=None,  # Default: Grow until pure or min_samples_split
                    min_samples_split=2,  # Default: Minimum samples to split a node
                    min_samples_leaf=1,  # Default: Minimum samples at a leaf
                    min_weight_fraction_leaf=0.0,  # Default: No minimum weight fraction
                    max_features="sqrt",  # Default: Square root of number of features
                    max_leaf_nodes=None,  # Default: No limit on leaf nodes
                    min_impurity_decrease=0.0,  # Default: No minimum impurity decrease
                    bootstrap=True,  # Default: Use bootstrap sampling
                    oob_score=False,  # Default: No out-of-bag score
                    n_jobs=None,  # Default: No parallel processing
                    random_state=42,  # Specified for reproducibility
                    verbose=0,  # Default: No verbose output
                    warm_start=False,  # Default: No warm start
                    class_weight=None,  # Default: No class weights
                    ccp_alpha=0.0,  # Default: No cost-complexity pruning
                    max_samples=None,  # Default: Use all samples in bootstrap
                ),
            ),
         
        ],
        voting="hard",  # Default: Majority voting
        weights=None,  # Default: Equal weights for all estimators
        n_jobs=None,  # Default: No parallel processing
        flatten_transform=True,  # Default: Flatten output for soft voting
        verbose=6,  # Default: No verbose output
    ),
}

# --- K-Fold setup ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=False)

# Dictionaries to store results
metrics_df_dict = {}  # Stores metrics table for each model
df_reordered_dict = {}  # Stores reordered dataset for each model
fold_indices_dict = {}  # Stores fold indices for each model

# --- Loop through models ---
for model_name, model in models.items():
    fold_metrics_list = []
    fold_indices_list = []

    # K-Fold CV
    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(
            y_test, y_pred, average="weighted"
        )  # 'weighted' for multi-class, adjust if binary

        # Save metrics
        fold_metrics_list.append(
            {"Fold": fold_index, "Accuracy": accuracy, "F1-Score": f1}
        )
        fold_indices_list.append({"train_idx": train_idx, "test_idx": test_idx})

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df_dict[model_name] = metrics_df
    fold_indices_dict[model_name] = fold_indices_list

    # Identify best fold based on F1-Score
    best_fold_idx = metrics_df[
        "F1-Score"
    ].idxmax()  # index of best fold (maximize F1-Score)
    best_test_idx = fold_indices_list[best_fold_idx]["test_idx"]

    # Reorder dataset: move best fold to the end
    remaining_idx = df.index.difference(best_test_idx)
    df_reordered = pd.concat([df.loc[remaining_idx], df.loc[best_test_idx]], axis=0)
    df_reordered_dict[model_name] = df_reordered

# --- OUTPUT ---
# metrics_df_dict[model_name] → fold metrics
# df_reordered_dict[model_name] → reordered dataset
