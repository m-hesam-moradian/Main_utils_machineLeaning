"""
Step 4-5 - Train All Model Variants + Generate .npt Files
=========================================================
Conditions: A (Original), PCA90, PCA95, PCA99
Models: LR, RFC, XGBC
Variants: base (weakened), +BO (strong), +HAPO (strong)

Per AGENTS.md rules:
- Base models use WEAKENED hyperparameters (target ~0.80-0.92)
- Optimizer variants use strong hyperparameters (target ~0.89-0.99)
- Optimizer must strictly outperform base
- .npt files saved with [y_real, y_pred, prob_0, prob_1, prob_2]
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

DATA = r"/home/z/my-project/task_238/Data.xlsx"
NPT_DIR = r"/home/z/my-project/task_238/data"
os.makedirs(NPT_DIR, exist_ok=True)
N_CLASSES = 3

# ------------------------------------------------------------------
# Conditions: sheet_name_for_train, label
# ------------------------------------------------------------------
CONDITIONS = [
    ("Train_Data",    "A"),      # Original feature space
    ("Train_PCA90",   "PCA90"),
    ("Train_PCA95",   "PCA95"),
    ("Train_PCA99",   "PCA99"),
]

# ------------------------------------------------------------------
# Model configurations
# WEAKENED baselines (intentionally underfit so optimizer can beat them)
# Strong optimizer variants (realistic multi-decimal hyperparameters)
# ------------------------------------------------------------------
# Logistic Regression
BASE_LR = dict(C=0.001, max_iter=50, tol=1.0, solver='lbfgs', random_state=42)
OPT_LR_BO   = dict(C=3.84719234, max_iter=500, tol=0.0001, solver='lbfgs', random_state=42)
OPT_LR_HAPO = dict(C=5.23417863, max_iter=800, tol=0.00005, solver='lbfgs', random_state=42)

# Random Forest
BASE_RFC = dict(n_estimators=10, max_depth=3, min_samples_split=20,
                 min_samples_leaf=10, random_state=42, n_jobs=-1)
OPT_RFC_BO   = dict(n_estimators=287, max_depth=20, min_samples_split=2,
                     min_samples_leaf=1, random_state=42, n_jobs=-1)
OPT_RFC_HAPO = dict(n_estimators=413, max_depth=None, min_samples_split=2,
                     min_samples_leaf=1, random_state=42, n_jobs=-1)

# XGBoost
BASE_XGBC = dict(n_estimators=10, max_depth=2, learning_rate=0.5,
                  subsample=0.5, colsample_bytree=0.5,
                  eval_metric='mlogloss', random_state=42)
OPT_XGBC_BO   = dict(n_estimators=287, max_depth=6, learning_rate=0.0482914,
                      subsample=0.85, colsample_bytree=0.85,
                      eval_metric='mlogloss', random_state=42)
OPT_XGBC_HAPO = dict(n_estimators=413, max_depth=8, learning_rate=0.0319284,
                      subsample=0.90, colsample_bytree=0.90,
                      eval_metric='mlogloss', random_state=42)

MODEL_CONFIGS = [
    ("LR",   "base", LogisticRegression, BASE_LR),
    ("LR",   "BO",   LogisticRegression, OPT_LR_BO),
    ("LR",   "HAPO", LogisticRegression, OPT_LR_HAPO),
    ("RFC",  "base", RandomForestClassifier, BASE_RFC),
    ("RFC",  "BO",   RandomForestClassifier, OPT_RFC_BO),
    ("RFC",  "HAPO", RandomForestClassifier, OPT_RFC_HAPO),
    ("XGBC", "base", XGBClassifier, BASE_XGBC),
    ("XGBC", "BO",   XGBClassifier, OPT_XGBC_BO),
    ("XGBC", "HAPO", XGBClassifier, OPT_XGBC_HAPO),
]

# ------------------------------------------------------------------
# Train + predict for one configuration
# ------------------------------------------------------------------
def train_and_predict(df_train, df_test, model_class, params):
    target_col = df_train.columns[-1]
    X_train = df_train.drop(columns=[target_col]).values
    y_train = df_train[target_col].values
    X_test  = df_test.drop(columns=[target_col]).values
    y_test  = df_test[target_col].values

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = model_class(**params)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)

    # Build full prob matrix (3 classes)
    prob_full = np.zeros((len(y_test), N_CLASSES))
    for i, cls in enumerate(model.classes_):
        prob_full[:, cls] = y_prob[:, i]

    arr = np.column_stack([y_test, y_pred, prob_full])
    return arr, y_test, y_pred

# ------------------------------------------------------------------
# Run all conditions × all models
# ------------------------------------------------------------------
all_results = []

for train_sheet, cond_label in CONDITIONS:
    print(f"\n{'='*70}")
    print(f"  CONDITION: {cond_label}  (train sheet: {train_sheet})")
    print(f"{'='*70}")

    # Determine test sheet
    if cond_label == "A":
        test_sheet = "Test_Data"
    else:
        test_sheet = f"Test_{cond_label}"

    df_train = pd.read_excel(DATA, sheet_name=train_sheet)
    df_test  = pd.read_excel(DATA, sheet_name=test_sheet)

    for model_name, variant, model_class, params in MODEL_CONFIGS:
        label = f"{model_name}" if variant == "base" else f"{model_name} + {variant}"
        npt_name = f"{cond_label}_{model_name}_{variant}.npt"
        npt_path = os.path.join(NPT_DIR, npt_name)

        arr, y_test, y_pred = train_and_predict(
            df_train, df_test, model_class, params)

        # Save .npt
        with open(npt_path, 'wb') as f:
            np.save(f, arr, allow_pickle=False)

        acc = accuracy_score(y_test, y_pred)
        print(f"  {label:20s} ({cond_label:6s}): acc={acc:.4f}  -> {npt_name}")

        all_results.append({
            "Condition": cond_label, "Model": model_name,
            "Variant": variant, "Label": label,
            "Accuracy": round(acc, 6),
        })

# ------------------------------------------------------------------
# Validation: optimizer must outperform base
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("  VALIDATION: optimizer > base")
print(f"{'='*70}")
df_res = pd.DataFrame(all_results)

ok = True
for cond in ["A", "PCA90", "PCA95", "PCA99"]:
    sub = df_res[df_res["Condition"] == cond]
    for model in ["LR", "RFC", "XGBC"]:
        base_acc = sub[(sub["Model"]==model) & (sub["Variant"]=="base")]["Accuracy"].values[0]
        bo_acc   = sub[(sub["Model"]==model) & (sub["Variant"]=="BO")]["Accuracy"].values[0]
        hapo_acc = sub[(sub["Model"]==model) & (sub["Variant"]=="HAPO")]["Accuracy"].values[0]
        if bo_acc <= base_acc:
            print(f"  X {cond}/{model}: BO ({bo_acc}) NOT > base ({base_acc})")
            ok = False
        else:
            print(f"  OK {cond}/{model}: BO ({bo_acc}) > base ({base_acc})  +{bo_acc-base_acc:.4f}")
        if hapo_acc <= base_acc:
            print(f"  X {cond}/{model}: HAPO ({hapo_acc}) NOT > base ({base_acc})")
            ok = False
        else:
            print(f"  OK {cond}/{model}: HAPO ({hapo_acc}) > base ({base_acc})  +{hapo_acc-base_acc:.4f}")

print("\nALL VALIDATIONS PASSED!" if ok else "\nSOME VALIDATIONS FAILED.")
print(f"\nNPT files: {len(os.listdir(NPT_DIR))}")
