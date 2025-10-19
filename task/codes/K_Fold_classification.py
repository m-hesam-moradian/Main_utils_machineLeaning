import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- Load dataset ---
excel_path = r"D:\ML\Main_utils_machineLeaning\task\BSE. No.14-Dataset.xlsx"
sheet_name = "DATA_Shuffled"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Anomaly_Detected"

# --- Ensure target is binary and not leaking ---
if df[target_column].nunique() > 2:
    median_value = df[target_column].median()
    df[target_column] = (df[target_column] > median_value).astype(int)

# --- Features and target ---
X = df.drop(columns=[target_column])
y = df[target_column]

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

models = {
    "SGDC": SGDClassifier(),
    "SVC": SVC(),
}

# --- K-Fold setup ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=False)

# --- Results containers ---
metrics_df_dict = {}
df_reordered_dict = {}
summary_df = []
df_prediction_dict = {}


# --- K-Fold loop ---
for model_name, model in models.items():
    fold_metrics = []
    fold_indices = []
    y_real_all = []
    y_pred_all = []

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Check for constant target in fold
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(
                f"⚠️ Skipping fold {fold_index} for {model_name}: constant target values."
            )
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save predictions
        y_real_all.extend(y_test)
        y_pred_all.extend(y_pred)

        # Save metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        fold_metrics.append({"Fold": fold_index, "Accuracy": acc, "F1-Score": f1})
        fold_indices.append({"train_idx": train_idx, "test_idx": test_idx})

    # Save metrics DataFrame
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df_dict[model_name] = metrics_df

    # Identify best fold
    best_fold_idx = metrics_df["F1-Score"].idxmax()
    best_test_idx = fold_indices[best_fold_idx]["test_idx"]
    remaining_idx = df.index.difference(best_test_idx)
    df_reordered = pd.concat([df.loc[remaining_idx], df.loc[best_test_idx]], axis=0)
    df_reordered_dict[model_name] = df_reordered

    # Add summary row
    summary_df.append(
        {
            "Model": model_name,
            "Best Fold": metrics_df.loc[best_fold_idx, "Fold"],
            "Best Accuracy": metrics_df.loc[best_fold_idx, "Accuracy"],
            "Best F1-Score": metrics_df.loc[best_fold_idx, "F1-Score"],
            "Mean Accuracy": metrics_df["Accuracy"].mean(),
            "Mean F1-Score": metrics_df["F1-Score"].mean(),
        }
    )

    # Save prediction DataFrame
    prediction_df = pd.DataFrame({"y_real": y_real_all, "y_pred": y_pred_all})
    df_prediction_dict[model_name] = prediction_df
# --- Save results to Excel ---
with pd.ExcelWriter(
    excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(
            writer, sheet_name=f"{model_name}_KFOLD_Metrics", index=False
        )
        df_reordered_dict[model_name].to_excel(
            writer, sheet_name=f"Data_after_KFold_{model_name}", index=False
        )
    pd.DataFrame(summary_df).to_excel(writer, sheet_name="Model_Summary", index=False)

print(
    f"✅ K-Fold results and summary added to '{excel_path}' with sheets for SVC, LGBC, and Model_Summary."
)
# --- Log fold results to console ---
print(f"\n📘 K-Fold Results for {model_name}:")
print(metrics_df.to_string(index=False, float_format="%.4f"))
best_fold = metrics_df.loc[best_fold_idx]
print(f"\n🏆 Best Fold for {model_name}: Fold {best_fold['Fold']}")
print(f"   Accuracy: {best_fold['Accuracy']:.4f}")
print(f"   F1-Score: {best_fold['F1-Score']:.4f}")
print(f"📊 Mean Accuracy: {metrics_df['Accuracy'].mean():.4f}")
print(f"📊 Mean F1-Score: {metrics_df['F1-Score'].mean():.4f}")
