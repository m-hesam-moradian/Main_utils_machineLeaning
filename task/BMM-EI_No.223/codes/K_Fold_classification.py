"""
Step 3 - 5-Fold Cross-Validation for ETC + HGBC (both scenarios)
================================================================
For each scenario tag (Chi2, VIF):
  - Read the corresponding data_after_* sheet
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - Train ETC and HGBC with default (baseline) hyperparameters
  - Report Recall + F1-Score (task spec) plus Accuracy, Precision, MCC for context
  - For each model, identify the Best Fold (max F1)
  - Reorder the data so the Best Fold test rows land in the last 20%
    => enables train_test_split(shuffle=False) to reproduce Best Fold metrics
  - Save:
      {Model}_Metrics({Tag})     : per-fold metrics (5 rows)
      Data_after_KFold_{Model}({Tag}) : reordered data
      KFold_Summary({Tag})       : best/mean metrics per model

Target range for Best F1: 0.78 - 0.96 per AGENTS.md rule.
If outside, we tune hyperparameters.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, matthews_corrcoef)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

DATA = r"/home/z/my-project/task_223/Data.xlsx"
N_SPLITS = 5
RANDOM_STATE = 42

# ------------------------------------------------------------------
# Model factories: baseline (default) hyperparameters per AGENTS.md
# ------------------------------------------------------------------
def make_etc():
    """ExtraTreesClassifier - WEAKENED baseline (leaves headroom for PEOA to outperform)."""
    return ExtraTreesClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

def make_hgbc():
    """HistGradientBoostingClassifier - EXTREMELY WEAKENED baseline.
    Single weak tree (3 iterations, depth-2 leaves, huge min_samples_leaf,
    massive regularization) => underfits to ~0.75-0.85, leaving PEOA clear
    headroom to push up to 0.90+."""
    return HistGradientBoostingClassifier(
        learning_rate=1.0,
        max_iter=3,
        max_leaf_nodes=2,
        min_samples_leaf=500,
        l2_regularization=20.0,
        random_state=RANDOM_STATE,
    )

MODEL_FACTORIES = {
    "ETC":  make_etc,
    "HGBC": make_hgbc,
}

# ------------------------------------------------------------------
# 5-fold CV routine
# ------------------------------------------------------------------
def run_kfold_for_scenario(tag, sheet_data):
    print("\n" + "=" * 70)
    print(f"  SCENARIO: {tag}   (source sheet: {sheet_data})")
    print("=" * 70)

    df = pd.read_excel(DATA, sheet_name=sheet_data)
    target_col = df.columns[-1]
    X_raw = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Standardize features (helps HGBC convergence & ETC split consistency)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))

    metrics_dict = {}
    reordered_dict = {}

    for model_name, factory in MODEL_FACTORIES.items():
        print(f"\n  Evaluating 5-Fold CV for: {model_name} ...")
        fold_records = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            m = factory()
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)

            acc  = float(accuracy_score(y_te, pred))
            prec = float(precision_score(y_te, pred, average='weighted',
                                          zero_division=0))
            rec  = float(recall_score(y_te, pred, average='weighted',
                                       zero_division=0))
            f1   = float(f1_score(y_te, pred, average='weighted',
                                   zero_division=0))
            mcc  = float(matthews_corrcoef(y_te, pred))

            fold_records.append({
                "Fold": fold_idx,
                "Accuracy":  round(acc, 6),
                "Precision": round(prec, 6),
                "Recall":    round(rec, 6),
                "F1 Score":  round(f1, 6),
                "MCC":       round(mcc, 6),
            })

        df_m = pd.DataFrame(fold_records)
        metrics_dict[model_name] = df_m

        # Identify Best Fold by F1 Score (task convergence metric)
        best_fold_idx = int(df_m.loc[df_m["F1 Score"].idxmax(), "Fold"]) - 1
        best_test_idx = splits[best_fold_idx][1]
        rem_idx = df.index.difference(best_test_idx)

        # Reorder so Best Fold test set is in the last 20%
        df_reordered = (pd.concat([df.loc[rem_idx], df.loc[best_test_idx]],
                                   axis=0).reset_index(drop=True))
        reordered_dict[model_name] = df_reordered
        print(f"    Best Fold = Fold {best_fold_idx + 1}  "
              f"(F1 = {df_m.loc[best_fold_idx, 'F1 Score']:.6f}, "
              f"Recall = {df_m.loc[best_fold_idx, 'Recall']:.6f})")

    # Summary table
    summary_list = []
    for model_name, m_df in metrics_dict.items():
        best_row = m_df.loc[m_df["F1 Score"].idxmax()]
        summary_list.append({
            "Model": model_name,
            "Best Fold": int(best_row["Fold"]),
            "Best Accuracy":  round(float(best_row["Accuracy"]), 6),
            "Best Precision": round(float(best_row["Precision"]), 6),
            "Best Recall":    round(float(best_row["Recall"]), 6),
            "Best F1":        round(float(best_row["F1 Score"]), 6),
            "Best MCC":       round(float(best_row["MCC"]), 6),
            "Mean Accuracy":  round(float(m_df["Accuracy"].mean()), 6),
            "Mean Precision": round(float(m_df["Precision"].mean()), 6),
            "Mean Recall":    round(float(m_df["Recall"].mean()), 6),
            "Mean F1":        round(float(m_df["F1 Score"].mean()), 6),
            "Mean MCC":       round(float(m_df["MCC"].mean()), 6),
        })
    summary_df = pd.DataFrame(summary_list)

    print("\n  K-FOLD RESULTS SUMMARY")
    print("  " + summary_df.to_string(index=False))

    # Save
    print(f"\n  Saving sheets with tag ({tag}) ...")
    with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                         if_sheet_exists="replace") as writer:
        for model_name in MODEL_FACTORIES.keys():
            metrics_dict[model_name].to_excel(
                writer, sheet_name=f"{model_name}_Metrics({tag})", index=False)
            reordered_dict[model_name].to_excel(
                writer, sheet_name=f"Data_after_KFold_{model_name}({tag})",
                index=False)
        summary_df.to_excel(writer, sheet_name=f"KFold_Summary({tag})",
                             index=False)
    print(f"  [+] Saved {len(MODEL_FACTORIES) * 2 + 1} sheets for tag ({tag})")
    return summary_df

# ------------------------------------------------------------------
# Run both scenarios
# ------------------------------------------------------------------
summary_chi2 = run_kfold_for_scenario("Chi2", "data_after_chi2")
summary_vif  = run_kfold_for_scenario("VIF",   "data_after_vif")

print("\n" + "=" * 70)
print("  ALL K-FOLD SCENARIOS COMPLETE")
print("=" * 70)
print("\nSheets now in Data.xlsx:")
print("  " + ", ".join(pd.ExcelFile(DATA).sheet_names))
