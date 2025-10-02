import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- Load dataset ---
sheet_name = "DATA"
df = pd.read_excel(
    r"D:\ML\Main_utils\task\136_Seismic_ETC_RTHA, BO.xlsx",
    sheet_name=sheet_name,
)
target_column = "Class"

# Convert target to categorical (binary classification example)
median_value = df[target_column].median()
df[target_column] = (df[target_column] > median_value).astype(int)

# Features and target
X = df.drop(columns=[target_column])
# --- Force all columns to numeric ---
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)  # Replace any conversion failures with 0
y = df[target_column]

# --- Define model (Extra Trees Classifier) ---
etc_model = ExtraTreesClassifier(
    n_estimators=200,  # number of trees
    criterion="gini",  # splitting criterion
    max_depth=None,  # no limit on tree depth
    min_samples_split=2,  # minimum samples to split
    min_samples_leaf=1,  # minimum samples per leaf
    max_features="sqrt",  # feature subset size
    bootstrap=False,  # no bootstrap (by default)
    random_state=42,  # reproducibility
    n_jobs=-1,  # use all cores
    verbose=0,
)

# --- K-Fold setup ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=False)

# Storage
fold_metrics_list = []
fold_indices_list = []

# --- K-Fold CV Loop ---
for fold_index, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit model
    etc_model.fit(X_train, y_train)
    y_pred = etc_model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Save metrics
    fold_metrics_list.append({"Fold": fold_index, "Accuracy": accuracy, "F1-Score": f1})
    fold_indices_list.append({"train_idx": train_idx, "test_idx": test_idx})

# --- Convert metrics to DataFrame ---
metrics_df = pd.DataFrame(fold_metrics_list)

# --- Cross-validation summary (mean ± std) ---
summary_metrics = {
    "Accuracy_mean": metrics_df["Accuracy"].mean(),
    "Accuracy_std": metrics_df["Accuracy"].std(),
    "F1_mean": metrics_df["F1-Score"].mean(),
    "F1_std": metrics_df["F1-Score"].std(),
}

# --- Identify best fold based on F1-Score ---
best_fold_idx = metrics_df["F1-Score"].idxmax()
best_test_idx = fold_indices_list[best_fold_idx]["test_idx"]

# --- Reorder dataset: move best fold to the end ---
remaining_idx = df.index.difference(best_test_idx)
df_reordered = pd.concat([df.loc[remaining_idx], df.loc[best_test_idx]], axis=0)

# --- OUTPUT ---
print("📊 Per-Fold Metrics (Extra Trees Classifier):")
print(metrics_df, "\n")

print("✅ Cross-Validation Summary (mean ± std):")
print(
    f"Accuracy: {summary_metrics['Accuracy_mean']:.4f} ± {summary_metrics['Accuracy_std']:.4f}"
)
print(f"F1-Score: {summary_metrics['F1_mean']:.4f} ± {summary_metrics['F1_std']:.4f}")
