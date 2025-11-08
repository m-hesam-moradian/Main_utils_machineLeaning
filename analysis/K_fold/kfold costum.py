import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# --- Load dataset ---
excel_path = r"C:\Users\Sam\Desktop\ML\204-Crop_Recommendation 2_HGBC__SVC_NO_DDOA_FS(chi2).xlsx"
sheet_name = "data for kfold"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = "Crop"

# --- Ensure target is binary ---
if df[target_column].nunique() > 2:
    median_value = df[target_column].median()
    df[target_column] = (df[target_column] > median_value).astype(int)

# --- Features and target ---
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Models ---
models = {

    "HGBC": HistGradientBoostingClassifier(),
    "SVC": SVC()
}

# --- K-Fold setup ---
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"⚠️ Skipping fold {fold_index} for {model_name}: constant target values.")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_real_all.extend(y_test)
        y_pred_all.extend(y_pred)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        fold_metrics.append({"Fold": fold_index, "Accuracy": acc, "F1-Score": f1})
        fold_indices.append({"train_idx": train_idx, "test_idx": test_idx})

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df_dict[model_name] = metrics_df

    best_fold_idx = metrics_df["F1-Score"].idxmax()
    best_test_idx = fold_indices[best_fold_idx]["test_idx"]
    remaining_idx = df.index.difference(best_test_idx)
    df_reordered = pd.concat([df.loc[remaining_idx], df.loc[best_test_idx]], axis=0)
    df_reordered_dict[model_name] = df_reordered

    summary_df.append({
        "Model": model_name,
        "Best Fold": metrics_df.loc[best_fold_idx, "Fold"],
        "Best Accuracy": metrics_df.loc[best_fold_idx, "Accuracy"],
        "Best F1-Score": metrics_df.loc[best_fold_idx, "F1-Score"],
        "Mean Accuracy": metrics_df["Accuracy"].mean(),
        "Mean F1-Score": metrics_df["F1-Score"].mean(),
    })

    prediction_df = pd.DataFrame({"y_real": y_real_all, "y_pred": y_pred_all})
    df_prediction_dict[model_name] = prediction_df

# --- Save results to Excel ---
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(writer, sheet_name=f"{model_name}_KFOLD_Metrics", index=False)
        df_reordered_dict[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}", index=False)
    pd.DataFrame(summary_df).to_excel(writer, sheet_name="Model_Summary", index=False)

print(f"✅ K-Fold results and summary added to '{excel_path}'.")

# --- Log and copy predictions ---
for model_name in models:
    print(f"\n📘 K-Fold Results for {model_name}:")
    print(metrics_df_dict[model_name].to_string(index=False, float_format="%.4f"))
    best_fold = metrics_df_dict[model_name].loc[metrics_df_dict[model_name]["F1-Score"].idxmax()]
    print(f"\n🏆 Best Fold for {model_name}: Fold {best_fold['Fold']}")
    print(f"   Accuracy: {best_fold['Accuracy']:.4f}")
    print(f"   F1-Score: {best_fold['F1-Score']:.4f}")
    print(f"📊 Mean Accuracy: {metrics_df_dict[model_name]['Accuracy'].mean():.4f}")
    print(f"📊 Mean F1-Score: {metrics_df_dict[model_name]['F1-Score'].mean():.4f}")

    # Copy predictions to clipboard
    df_prediction_dict[model_name].to_clipboard(index=False)
    print(f"📋 Predictions for {model_name} copied to clipboard.")