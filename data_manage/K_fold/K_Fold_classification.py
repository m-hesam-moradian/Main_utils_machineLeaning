# D:\ML\Project-2\src\model\train_model.py

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Path & target
DATA_PATH = "D:\ML\main_structure\data\data.xlsx"
TARGET = "HVAC Efficiency"

# Load dataset
df = pd.read_excel(DATA_PATH, sheet_name="data")

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Define model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)

# KFold setup
kf = KFold(n_splits=5)

# Collect results
fold_results = []
test_indices_best_fold = None
best_accuracy = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    fold_results.append({"Fold": fold, "Accuracy": acc})
    print(f"Fold {fold} → Accuracy: {acc:.4f}")

    # Track the best fold
    if acc > best_accuracy:
        best_accuracy = acc
        test_indices_best_fold = val_idx

# Convert results to DataFrame
fold_scores_df = pd.DataFrame(fold_results)

# Reorder the original dataset: best fold test data at the bottom
X_test_best = X.iloc[test_indices_best_fold].copy()
y_test_best = y.iloc[test_indices_best_fold].copy()

# Remaining data (all except best fold test)
remaining_indices = X.index.difference(test_indices_best_fold)
X_remaining = X.iloc[remaining_indices].copy()
y_remaining = y.iloc[remaining_indices].copy()

# Combine into dataafterkfold
dataafterkfold = pd.concat([X_remaining, X_test_best], axis=0).reset_index(drop=True)
dataafterkfold[TARGET] = pd.concat([y_remaining, y_test_best], axis=0).reset_index(
    drop=True
)

print("\n✅ K-Fold completed")
print("Fold Scores:\n", fold_scores_df)
print(
    f"\n🏆 Best Fold: {fold_scores_df['Fold'][fold_scores_df['Accuracy'].idxmax()]} "
    f"with Accuracy: {best_accuracy:.4f}"
)

# Now `dataafterkfold` contains all original data with the best fold's test set at the bottom
