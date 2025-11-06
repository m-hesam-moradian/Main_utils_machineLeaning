import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier

# --- Load dataset ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.24-.xlsx"
sheet_name = "DATA_Shuffled"
target_column = "Fault_Status"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Define models ---
models = {
    "LR": LogisticRegression(max_iter=1000),
    "ETC": ExtraTreesClassifier(),
    "ADAC": AdaBoostClassifier()
}

# --- K-Fold setup ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

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

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"⚠️ Skipping fold {fold_index} for {model_name}: constant target values.")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_real_all.extend(y_test)
        y_pred_all.extend(y_pred)

        f1 = f1_score(y_test, y_pred, average="binary")
        precision = precision_score(y_test, y_pred, average="binary")
        fold_metrics.append({"Fold": fold_index, "F1-Score": f1, "Precision": precision})
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
        "Best F1-Score": metrics_df.loc[best_fold_idx, "F1-Score"],
        "Best Precision": metrics_df.loc[best_fold_idx, "Precision"],
        "Mean F1-Score": metrics_df["F1-Score"].mean(),
        "Mean Precision": metrics_df["Precision"].mean(),
    })

    prediction_df = pd.DataFrame({"y_real": y_real_all, "y_pred": y_pred_all})
    df_prediction_dict[model_name] = prediction_df

# --- Save results to Excel ---
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for model_name in models:
        metrics_df_dict[model_name].to_excel(writer, sheet_name=f"{model_name}_KFOLD_Metrics", index=False)
        df_reordered_dict[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}", index=False)
    pd.DataFrame(summary_df).to_excel(writer, sheet_name="Model_Summary", index=False)

# --- Log fold results to console ---
for model_name in models:
    metrics_df = metrics_df_dict[model_name]
    best_fold_idx = metrics_df["F1-Score"].idxmax()
    best_fold = metrics_df.loc[best_fold_idx]

    print(f"\n📘 K-Fold Results for {model_name}:")
    print(metrics_df.to_string(index=False, float_format="%.4f"))
    print(f"\n🏆 Best Fold for {model_name}: Fold {best_fold['Fold']}")
    print(f"   F1-Score: {best_fold['F1-Score']:.4f}")
    print(f"   Precision: {best_fold['Precision']:.4f}")
    print(f"📊 Mean F1-Score: {metrics_df['F1-Score'].mean():.4f}")
    print(f"📊 Mean Precision: {metrics_df['Precision'].mean():.4f}")

# --- Copy predictions to clipboard (change model name if needed) ---
df_prediction_dict["LR"].to_clipboard(index=False)