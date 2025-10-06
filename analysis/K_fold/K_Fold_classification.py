import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- Load dataset ---
sheet_name = "Data after K-FOLD (votingC)"
excel_path = r"D:\ML\Main_utils\task\WA_Fn-UseC_-HR-Employee-Attrition.xlsx"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Attrition"

# --- Convert target to binary if needed ---
if df[target_column].dtype != "int":
    median_value = df[target_column].median()
    df[target_column] = (df[target_column] > median_value).astype(int)

# --- Features and target ---
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Define models ---
models = {
    "SVC": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    "LGBC": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

# --- K-Fold setup ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# --- Results containers ---
metrics_df_dict = {}
df_reordered_dict = {}

# --- K-Fold loop ---
for model_name, model in models.items():
    fold_metrics = []
    fold_indices = []

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        fold_metrics.append({"Fold": fold_index, "Accuracy": acc, "F1-Score": f1})
        fold_indices.append({"train_idx": train_idx, "test_idx": test_idx})

    # Save metrics
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df_dict[model_name] = metrics_df

    # Reorder dataset based on best fold
    best_fold_idx = metrics_df["F1-Score"].idxmax()
    best_test_idx = fold_indices[best_fold_idx]["test_idx"]
    remaining_idx = df.index.difference(best_test_idx)
    df_reordered = pd.concat([df.loc[remaining_idx], df.loc[best_test_idx]], axis=0)
    df_reordered_dict[model_name] = df_reordered

# --- Save results to Excel ---
output_path = r"D:\ML\Main_utils\task\KFold_Results_SVC_LGBC.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(
            writer, sheet_name=f"{model_name}_Metrics", index=False
        )
        df_reordered_dict[model_name].to_excel(
            writer, sheet_name=f"Data_after_KFold_{model_name}", index=False
        )

print(f"✅ K-Fold results saved to '{output_path}' with sheets for SVC and LGBC.")
