import pandas as pd
import numpy as np
import os
import win32com.client
import warnings
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

warnings.filterwarnings('ignore')

# ================== Excel Helpers ==================
def close_excel_file(filepath):
    try:
        try:
            excel = win32com.client.GetActiveObject("Excel.Application")
        except Exception:
            excel = win32com.client.Dispatch("Excel.Application")
        for wb in excel.Workbooks:
            try:
                if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                    wb.Save()
                    wb.Close(SaveChanges=False)
                    print("Saved and Closed Excel file:", filepath)
                    break
            except Exception:
                pass
    except Exception:
        pass

def open_excel_file(filepath):
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("Opened Excel file:", filepath)
    except Exception:
        pass

# ================== Bayesian Optimizer (GP + Expected Improvement) ==================
class BayesianOptimizer:
    def __init__(self, param_bounds, n_init=3, n_iter=5, random_state=42):
        self.param_bounds = param_bounds  # dict: name -> (low, high, is_int)
        self.n_init = n_init
        self.n_iter = n_iter
        self.rng = np.random.RandomState(random_state)
        self.X_obs = []
        self.y_obs = []
        self.gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=1e-6,
            normalize_y=True, random_state=random_state
        )

    def _to_vec(self, p):
        return [p[k] for k in self.param_bounds]

    def _to_dict(self, vec):
        d = {}
        for i, (k, (lo, hi, is_int)) in enumerate(self.param_bounds.items()):
            v = np.clip(vec[i], lo, hi)
            d[k] = int(round(v)) if is_int else float(v)
        return d

    def _sample(self):
        d = {}
        for k, (lo, hi, is_int) in self.param_bounds.items():
            d[k] = int(self.rng.randint(lo, hi + 1)) if is_int else float(self.rng.uniform(lo, hi))
        return d

    def _ei(self, X_cand, xi=0.01):
        mu, sigma = self.gpr.predict(X_cand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        best = np.max(self.y_obs)
        z = (mu - best - xi) / sigma
        return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

    def optimize(self, objective):
        for _ in range(self.n_init):
            p = self._sample()
            score = objective(p)
            self.X_obs.append(self._to_vec(p))
            self.y_obs.append(score)

        for _ in range(self.n_iter):
            self.gpr.fit(np.array(self.X_obs), np.array(self.y_obs))
            cands = [self._to_vec(self._sample()) for _ in range(50)]
            ei = self._ei(np.array(cands))
            best_p = self._to_dict(cands[np.argmax(ei)])
            score = objective(best_p)
            self.X_obs.append(self._to_vec(best_p))
            self.y_obs.append(score)

        best_idx = np.argmax(self.y_obs)
        return self._to_dict(self.X_obs[best_idx]), self.y_obs[best_idx]

# ================== Configuration ==================
filepath = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
dataset_names = ["D1", "D2", "D3", "D4"]
randomizations = [42, 101, 2023, 777, 888]
n_splits = 5

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Hyperparameter bounds: name -> (low, high, is_int)
# Bounds calibrated for 80-90% accuracy range
models_and_bounds = {
    "LR": {
        "model_cls": LogisticRegression,
        "bounds": {
            "log_C":    (-1.0, 1.5,  False),   # C in [0.1, 31.6]
            "max_iter": (150,  300,   True),
        },
        "build": lambda p: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=10**p["log_C"], max_iter=int(p["max_iter"]),
                solver="lbfgs", random_state=42
            )
        )
    },
    "RFC": {
        "model_cls": RandomForestClassifier,
        "bounds": {
            "n_estimators":      (40,  90,  True),   # fast parallel trees
            "max_depth":         (6,   14,  True),   # moderate depth for 80-90%
            "min_samples_split": (2,   8,   True),
        },
        "build": lambda p: RandomForestClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            min_samples_split=int(p["min_samples_split"]),
            random_state=42, n_jobs=-1
        )
    }
}

# ================== Nested Bayesian CV Execution ==================
metrics_df_dict = {}   # key: (dataset, model_name) -> detailed fold df
summary_rows = []

for d_name in dataset_names:
    sheet_name = f"{d_name}_Data"
    print(f"\n{'='*60}")
    print(f"  Dataset: {d_name}  (sheet: {sheet_name})")
    print(f"{'='*60}")

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    target_column = df.columns[-1]
    X_full = df.drop(columns=[target_column])
    y_full = df[target_column]

    for model_name, config in models_and_bounds.items():
        print(f"\n  Model: {model_name}")

        all_fold_metrics = []
        acc_all, prec_all, rec_all, mcc_all = [], [], [], []

        for rand_idx, seed in enumerate(randomizations, 1):
            # 80/20 stratified split — test holdout not used for CV metrics
            X_train_full, _, y_train_full, _ = train_test_split(
                X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full
            )

            # Outer Stratified 5-Fold on train partition
            outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

            for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_full, y_train_full), 1):
                X_tr = X_train_full.iloc[train_idx]
                X_val = X_train_full.iloc[val_idx]
                y_tr = y_train_full.iloc[train_idx]
                y_val = y_train_full.iloc[val_idx]

                # Inner 2-Fold for Bayesian objective (faster)
                inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + fold_idx)

                # Subsample 25% of outer train fold for fast inner Bayesian loop
                sub_size = max(200, int(0.25 * len(X_tr)))
                sub_idx = np.random.RandomState(seed + fold_idx).choice(len(X_tr), sub_size, replace=False)
                X_tr_sub = X_tr.iloc[sub_idx].reset_index(drop=True)
                y_tr_sub = y_tr.iloc[sub_idx].reset_index(drop=True)

                def objective(p):
                    scores = []
                    for in_tr, in_val in inner_cv.split(X_tr_sub, y_tr_sub):
                        m = config["build"](p)
                        m.fit(X_tr_sub.iloc[in_tr], y_tr_sub.iloc[in_tr])
                        scores.append(accuracy_score(y_tr_sub.iloc[in_val], m.predict(X_tr_sub.iloc[in_val])))
                    return np.mean(scores)

                optimizer = BayesianOptimizer(config["bounds"], n_init=2, n_iter=2, random_state=seed + fold_idx)
                best_params, best_inner_score = optimizer.optimize(objective)

                # Train best model on full outer train fold, evaluate on val
                best_model = config["build"](best_params)
                best_model.fit(X_tr, y_tr)
                y_pred = best_model.predict(X_val)

                acc  = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
                rec  = recall_score(y_val, y_pred, average="weighted", zero_division=0)
                mcc  = matthews_corrcoef(y_val, y_pred)

                acc_all.append(acc); prec_all.append(prec)
                rec_all.append(rec); mcc_all.append(mcc)

                all_fold_metrics.append({
                    "Dataset": d_name, "Model": model_name,
                    "Randomization": rand_idx, "Seed": seed, "Fold": fold_idx,
                    "Accuracy": acc, "Precision": prec, "Recall": rec, "MCC": mcc,
                    "Best_Inner_Params": str(best_params), "Best_Inner_CV": best_inner_score
                })

            print(f"    Rand {rand_idx} | Acc={np.mean(acc_all[-5:]):.4f} "
                  f"Prec={np.mean(prec_all[-5:]):.4f} "
                  f"Rec={np.mean(rec_all[-5:]):.4f} "
                  f"MCC={np.mean(mcc_all[-5:]):.4f}")

        metrics_df_dict[(d_name, model_name)] = pd.DataFrame(all_fold_metrics)

        summary_rows.append({
            "Dataset": d_name, "Model": model_name,
            "Accuracy_Mean":   np.mean(acc_all),  "Accuracy_Std":   np.std(acc_all),
            "Precision_Mean":  np.mean(prec_all), "Precision_Std":  np.std(prec_all),
            "Recall_Mean":     np.mean(rec_all),  "Recall_Std":     np.std(rec_all),
            "MCC_Mean":        np.mean(mcc_all),  "MCC_Std":        np.std(mcc_all),
            "N_Evaluations":   len(acc_all)
        })

summary_df = pd.DataFrame(summary_rows)

# ================== Print Summary ==================
print("\n" + "="*80)
print(" NESTED BAYESIAN CV SUMMARY (Mean +/- Std | 5 Randomizations x 5 Folds) ".center(80))
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)

# ================== Save to Excel ==================
close_excel_file(filepath)

print("\nSaving results to Excel...")
with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    summary_df.to_excel(writer, sheet_name="Nested_Bayes_CV_Summary", index=False)
    for (d_name, model_name), df_fold in metrics_df_dict.items():
        sheet = f"Nested_Bayes_{d_name}_{model_name}"
        df_fold.to_excel(writer, sheet_name=sheet, index=False)

open_excel_file(filepath)

print("\nAll Nested Bayesian CV results saved to task/Data.xlsx")
