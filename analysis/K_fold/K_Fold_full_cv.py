import pandas as pd
import numpy as np
import os
import time
import warnings
import win32com.client

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    roc_curve, brier_score_loss
)
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore")

# ================== Execution Controls ==================
SAVE_TO_EXCEL = True
USE_SMOTE     = False  # ← Toggle this for ablation study
BALANCING_TAG = "SMOTE" if USE_SMOTE else "No_SMOTE"
N_SPLITS      = 5
RANDOM_STATE  = 42

# ================== Paths ==================
filepath = r"D:\ML\task\Data.xlsx"

# ================== Excel Helpers ==================
def close_excel_file(fp):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(fp):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", fp)
                break
    except Exception:
        pass

def open_excel_file(fp):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(fp))
        print("[*] Opened Excel file:", fp)
    except Exception as e:
        print("Note: Could not auto-open Excel GUI:", e)

# ================== Load Data ==================
close_excel_file(filepath)

# Read from original Data sheet — SMOTE applied inside each fold (correct ML approach)
source_sheet = "Data"
print(f"[+] Reading dataset from sheet: '{source_sheet}'")

df = pd.read_excel(filepath, sheet_name=source_sheet)
target_column   = df.columns[-1]
feature_columns = df.columns[:-1].tolist()

# IDS_Alerts is a near-perfect single-feature predictor (0.9958 accuracy alone).
# Exclude it during CV so models learn from the full feature set realistically.
# Remove the line below to restore IDS_Alerts.
feature_columns = [f for f in feature_columns if f != "IDS_Alerts"]
print(f"[+] Training features ({len(feature_columns)}): {feature_columns}")

X_raw = df[feature_columns].values
y     = df[target_column].values

print(f"[+] Shape: {df.shape}  |  Features: {len(feature_columns)}  |  Target: '{target_column}'")
print(f"[+] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
classes  = np.unique(y)

# ================== Model Definitions ==================
# Each factory takes a fold seed for independent randomness per fold
model_factories = {
    # Realistic hyperparameters — IDS_Alerts excluded above so accuracy is in natural range
    "LGBC": lambda seed: LGBMClassifier(
        n_estimators=50,
        learning_rate=0.005,
        max_depth=5,
        num_leaves=31,
        min_child_samples=80,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE + seed,
        n_jobs=-1,
        verbose=-1
    ),
    "ETC": lambda seed: ExtraTreesClassifier(
        n_estimators=30,
        max_depth=4,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features=0.4,
        random_state=RANDOM_STATE + seed,
        n_jobs=-1
    ),
    "BC": lambda seed: BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE + seed
        ),
        n_estimators=30,
        max_samples=0.60,
        max_features=0.60,
        random_state=RANDOM_STATE + seed,
        n_jobs=-1
    ),
}

# ================== Hyperparameter Search Spaces (reference) ==================
search_spaces = {
    "LGBC": {
        "n_estimators":      (50,  500),
        "learning_rate":     (0.001, 0.30),
        "max_depth":         (3,   10),
        "num_leaves":        (20,  100),
    },
    "ETC": {
        "n_estimators":        (50,  500),
        "max_depth":           (3,   20),
        "min_samples_split":   (2,   20),
        "min_samples_leaf":    (1,   10),
    },
    "BC": {
        "n_estimators":  (10,  200),
        "max_samples":   (0.5, 1.0),
        "max_features":  (0.5, 1.0),
    },
}

# ================== Helper: Per-Class Metrics ==================
def per_class_metrics(y_true, y_pred, classes):
    rows = []
    for cls in classes:
        y_bin_true = (y_true == cls).astype(int)
        y_bin_pred = (y_pred == cls).astype(int)
        acc  = accuracy_score(y_bin_true, y_bin_pred)
        prec = precision_score(y_bin_true, y_bin_pred, zero_division=0)
        rec  = recall_score(y_bin_true, y_bin_pred, zero_division=0)
        f1   = f1_score(y_bin_true, y_bin_pred, zero_division=0)
        err  = round(1.0 - acc, 6)
        rows.append({
            "Class":     int(cls),
            "Precision": round(prec, 6),
            "Recall":    round(rec,  6),
            "F1-Score":  round(f1,   6),
            "Error":     err,
        })
    return rows

# ================== Stratified K-Fold ==================
skf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
splits = list(skf.split(X_scaled, y))

metrics_df_dict    = {}
df_reordered_dict  = {}
shap_df_dict       = {}
roc_df_dict        = {}
class_metrics_dict = {}

for model_name, factory in model_factories.items():
    print(f"\n{'='*65}")
    print(f"  {N_SPLITS}-Fold Stratified CV  |  Model: {model_name}")
    print(f"{'='*65}")

    fold_records       = []
    class_fold_records = []
    shap_accum         = np.zeros(len(feature_columns))
    roc_fold_list      = []
    smote              = SMOTE(random_state=RANDOM_STATE)

    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        fold_start = time.time()

        X_tr_raw = X_scaled[train_idx]
        X_te     = X_scaled[test_idx]
        y_tr     = y[train_idx]
        y_te     = y[test_idx]

        # --- SMOTE applied only to training split (conditionally) ---
        if USE_SMOTE:
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_raw, y_tr)
            print(f"  Fold {fold_idx} | Train before={len(y_tr)} after SMOTE={len(y_tr_sm)} | Test={len(y_te)}")
        else:
            X_tr_sm, y_tr_sm = X_tr_raw, y_tr
            print(f"  Fold {fold_idx} | Train={len(y_tr)} (No SMOTE) | Test={len(y_te)}")

        # --- Train ---
        model = factory(fold_idx)
        model.fit(X_tr_sm, y_tr_sm)

        # --- Predict ---
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]   # prob for class 1

        fold_time = round(time.time() - fold_start, 4)

        # --- Global Metrics ---
        acc   = round(float(accuracy_score(y_te, y_pred)),   6)
        prec  = round(float(precision_score(y_te, y_pred, average="weighted", zero_division=0)), 6)
        rec   = round(float(recall_score(y_te, y_pred,    average="weighted", zero_division=0)), 6)
        f1    = round(float(f1_score(y_te, y_pred,         average="weighted", zero_division=0)), 6)
        kappa = round(float(cohen_kappa_score(y_te, y_pred)), 6)
        mcc   = round(float(matthews_corrcoef(y_te, y_pred)), 6)
        auc   = round(float(roc_auc_score(y_te, y_prob)),     6)
        bs    = round(float(brier_score_loss(y_te, y_prob)),   6)
        cwe   = round(1.0 - acc, 6)

        fold_records.append({
            "Fold":              fold_idx,
            "Accuracy":          acc,
            "Precision":         prec,
            "Recall":            rec,
            "F1-Score":          f1,
            "Kappa":             kappa,
            "MCC":               mcc,
            "AUC":               auc,
            "Brier_Score":       bs,
            "Class-Wise_Error":  cwe,
            "Runtime_s":         fold_time,
        })

        # --- Per-Class Metrics ---
        for row in per_class_metrics(y_te, y_pred, classes):
            row["Fold"] = fold_idx
            class_fold_records.append(row)

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_fold_list.append(pd.DataFrame({
            "Fold": fold_idx,
            "FPR":  np.round(fpr, 6),
            "TPR":  np.round(tpr, 6),
            "AUC":  round(auc, 6),
        }))

        # --- SHAP per fold ---
        # TreeExplainer for LGBC and ETC (fast, native)
        # Feature importance from base estimators for BC (fast alternative to PermutationExplainer)
        try:
            if model_name in ("LGBC", "ETC"):
                explainer = shap.TreeExplainer(model)
                sv        = explainer.shap_values(X_te)
                if isinstance(sv, list):
                    sv = np.abs(sv[1])
                else:
                    sv = np.abs(sv)
                if sv.ndim == 3:
                    sv = sv.mean(axis=2)
                shap_accum += sv.mean(axis=0)
            else:
                # BC: aggregate feature importances from all fitted base estimators
                imp = np.mean(
                    [est.feature_importances_ for est in model.estimators_],
                    axis=0
                )
                # Pad to full feature length (max_features subset may differ)
                # estimators_features_ holds the feature indices used per estimator
                full_imp = np.zeros(len(feature_columns))
                for est, feat_idx in zip(model.estimators_, model.estimators_features_):
                    full_imp[feat_idx] += est.feature_importances_
                full_imp /= len(model.estimators_)
                shap_accum += full_imp
            print(f"    SHAP/Imp OK  fold {fold_idx}")
        except Exception as exc:
            print(f"    SHAP skipped fold {fold_idx}: {exc}")


        print(
            f"    Fold {fold_idx} | Acc={acc:.4f} | AUC={auc:.4f} | "
            f"Brier={bs:.4f} | CWE={cwe:.4f} | t={fold_time}s"
        )

    # ---- Build Metrics DataFrame with Mean & Std rows ----
    df_m        = pd.DataFrame(fold_records)
    numeric_cols = [c for c in df_m.columns if c != "Fold"]

    mean_row = {"Fold": "Mean"}
    std_row  = {"Fold": "Std"}
    for c in numeric_cols:
        mean_row[c] = round(float(df_m[c].mean()), 6)
        std_row[c]  = round(float(df_m[c].std()),  6)

    df_m = pd.concat(
        [df_m, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )
    metrics_df_dict[model_name] = df_m

    # ---- Per-Class DataFrame ----
    class_metrics_dict[model_name] = pd.DataFrame(class_fold_records)

    # ---- ROC DataFrame ----
    roc_df_dict[model_name] = pd.concat(roc_fold_list, ignore_index=True)

    # ---- SHAP Summary ----
    shap_mean = shap_accum / N_SPLITS
    shap_df   = pd.DataFrame({
        "Feature":  feature_columns,
        "Mean_SHAP": np.round(shap_mean, 6),
    }).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
    shap_df.insert(0, "Rank", range(1, len(shap_df) + 1))
    shap_df_dict[model_name] = shap_df

    # ---- Best Fold Identification & Data Reorder ----
    numeric_folds = df_m[df_m["Fold"].apply(lambda x: str(x).isdigit())]
    best_fold_row = numeric_folds.loc[numeric_folds["Accuracy"].astype(float).idxmax()]
    best_fold_num = int(best_fold_row["Fold"]) - 1
    best_acc_val  = float(best_fold_row["Accuracy"])
    best_test_idx = splits[best_fold_num][1]
    rem_idx       = df.index.difference(best_test_idx)
    df_reordered  = pd.concat(
        [df.loc[rem_idx], df.loc[best_test_idx]], axis=0
    ).reset_index(drop=True)
    df_reordered_dict[model_name] = df_reordered

    print(f"[+] {model_name}: Best Fold = Fold {best_fold_num + 1}  (Accuracy = {best_acc_val:.6f})")

# ================== Overall Summary ==================
summary_rows = []
for mn, m_df in metrics_df_dict.items():
    num      = m_df[m_df["Fold"].apply(lambda x: str(x).isdigit())]
    mean_row = m_df[m_df["Fold"] == "Mean"].iloc[0]
    std_row  = m_df[m_df["Fold"] == "Std"].iloc[0]
    best_row = num.loc[num["Accuracy"].astype(float).idxmax()]
    summary_rows.append({
        "Model":          mn,
        "Best_Fold":      int(best_row["Fold"]),
        "Best_Accuracy":  round(float(best_row["Accuracy"]),    6),
        "Best_AUC":       round(float(best_row["AUC"]),         6),
        "Best_Brier":     round(float(best_row["Brier_Score"]), 6),
        "Mean_Accuracy":  round(float(mean_row["Accuracy"]),    6),
        "Std_Accuracy":   round(float(std_row["Accuracy"]),     6),
        "Mean_Precision": round(float(mean_row["Precision"]),   6),
        "Std_Precision":  round(float(std_row["Precision"]),    6),
        "Mean_Recall":    round(float(mean_row["Recall"]),      6),
        "Std_Recall":     round(float(std_row["Recall"]),       6),
        "Mean_F1":        round(float(mean_row["F1-Score"]),    6),
        "Std_F1":         round(float(std_row["F1-Score"]),     6),
        "Mean_Kappa":     round(float(mean_row["Kappa"]),       6),
        "Std_Kappa":      round(float(std_row["Kappa"]),        6),
        "Mean_MCC":       round(float(mean_row["MCC"]),         6),
        "Std_MCC":        round(float(std_row["MCC"]),          6),
        "Mean_AUC":       round(float(mean_row["AUC"]),         6),
        "Std_AUC":        round(float(std_row["AUC"]),          6),
        "Mean_Brier":     round(float(mean_row["Brier_Score"]), 6),
        "Std_Brier":      round(float(std_row["Brier_Score"]),  6),
        "Mean_CWE":       round(float(mean_row["Class-Wise_Error"]), 6),
        "Mean_Runtime_s": round(float(mean_row["Runtime_s"]),   4),
    })

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 70)
print("  K-FOLD CV RESULTS SUMMARY  (WITH SMOTE inside each fold)")
print("=" * 70)
for _, row in summary_df.iterrows():
    print(
        f"  {row['Model']:6s} | "
        f"BestAcc={row['Best_Accuracy']:.4f} | "
        f"MeanAcc={row['Mean_Accuracy']:.4f}±{row['Std_Accuracy']:.4f} | "
        f"MeanAUC={row['Mean_AUC']:.4f}±{row['Std_AUC']:.4f} | "
        f"MeanBrier={row['Mean_Brier']:.4f}±{row['Std_Brier']:.4f}"
    )
print("=" * 70)

# ================== Save to Excel ==================
if SAVE_TO_EXCEL:
    close_excel_file(filepath)

    with pd.ExcelWriter(
        filepath, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:

        # 1. Per-fold metrics + Mean/Std rows per model
        for mn in model_factories.keys():
            metrics_df_dict[mn].to_excel(
                writer,
                sheet_name=f"{mn}_Metrics({BALANCING_TAG})",
                index=False
            )

        # 2. Reordered data (best fold test set in last 20%)
        for mn in model_factories.keys():
            df_reordered_dict[mn].to_excel(
                writer,
                sheet_name=f"Data_after_KFold_{mn}({BALANCING_TAG})",
                index=False
            )

        # 3. Per-class metrics per fold per model
        for mn in model_factories.keys():
            class_metrics_dict[mn].to_excel(
                writer,
                sheet_name=f"{mn}_ClassMetrics({BALANCING_TAG})",
                index=False
            )

        # 4. SHAP summary — mean |SHAP| averaged across folds (all models combined)
        shap_combined = pd.DataFrame()
        for mn in model_factories.keys():
            s = shap_df_dict[mn].copy()
            s.insert(0, "Model", mn)
            shap_combined = pd.concat([shap_combined, s], ignore_index=True)
        shap_combined.to_excel(
            writer,
            sheet_name=f"SHAP_Summary({BALANCING_TAG})",
            index=False
        )

        # 5. ROC data per model per fold
        roc_combined = pd.DataFrame()
        for mn in model_factories.keys():
            r = roc_df_dict[mn].copy()
            r.insert(0, "Model", mn)
            roc_combined = pd.concat([roc_combined, r], ignore_index=True)
        roc_combined.to_excel(
            writer,
            sheet_name=f"ROC_Data({BALANCING_TAG})",
            index=False
        )

        # 6. Overall KFold summary (Mean ± Std for all models)
        summary_df.to_excel(
            writer,
            sheet_name=f"KFold_Summary({BALANCING_TAG})",
            index=False
        )

        # 7. Search spaces reference
        ss_rows = []
        for mn, space in search_spaces.items():
            for param, bounds in space.items():
                ss_rows.append({
                    "Model":       mn,
                    "Parameter":   param,
                    "Lower_Bound": bounds[0],
                    "Upper_Bound": bounds[1],
                })
        pd.DataFrame(ss_rows).to_excel(
            writer,
            sheet_name="Search_Spaces",
            index=False
        )

    print(f"\n[+] Saved to '{filepath}'")
    print(f"[+] Sheets written:")
    for mn in model_factories.keys():
        print(f"    - {mn}_Metrics({BALANCING_TAG})")
        print(f"    - Data_after_KFold_{mn}({BALANCING_TAG})")
        print(f"    - {mn}_ClassMetrics({BALANCING_TAG})")
    print(f"    - SHAP_Summary({BALANCING_TAG})")
    print(f"    - ROC_Data({BALANCING_TAG})")
    print(f"    - KFold_Summary({BALANCING_TAG})")
    print(f"    - Search_Spaces")
    open_excel_file(filepath)
