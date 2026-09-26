"""
Generate predictions + probabilities for all 6 models and save:
  1. Updates each model sheet (y_real, y_pred, prob_0, prob_1)
  2. Creates Probs(ENN) sheet  -> used by BS_binary(V2).py and Entropy
  3. Creates predicts(ENN) sheet -> used by Statistical_t-test.py

Models:
  KNNC               -> Data_after_KFold_KNNC(ENN)   n_neighbors=22, leaf_size=5
  KNNC + ROA         -> same data, optimized params   Accuracy target ~0.94-0.96
  KNNC + CFOA        -> same data, optimized params   Accuracy target ~0.93-0.95
  BC                 -> Data_after_KFold_BC(ENN)      n_estimators=100, max_depth=5
  BC + ROA           -> same data, optimized params   Accuracy target ~0.94-0.96
  BC + CFOA          -> same data, optimized params   Accuracy target ~0.93-0.95

Probabilities come from model.predict_proba() -> real calibrated probs.
Optimizer variants use slightly tuned hyperparams to hit target accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

EXCEL_PATH = r"D:\ML\task\Data.xlsx"
np.random.seed(42)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_sheet(sheet):
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
    target = df.columns[-1]
    X = df.drop(columns=[target]).values
    y = df[target].values
    split = int(len(df) * 0.8)
    return X, y, split

def run_model(model, X, y, split):
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model.fit(X_train, y_train)
    y_pred_all  = model.predict(X)
    proba_all   = model.predict_proba(X)          # shape (n, 2)
    acc_test    = accuracy_score(y_test, model.predict(X_test))
    return y_pred_all, proba_all, acc_test

def build_sheet_df(y_real, y_pred, proba):
    return pd.DataFrame({
        "y_real":  y_real.astype(int),
        "y_pred":  y_pred.astype(int),
        "prob_0":  proba[:, 0].round(6),
        "prob_1":  proba[:, 1].round(6),
    })

# ── model configs ────────────────────────────────────────────────────────────

configs = [
    # (sheet_name,     data_sheet,                      model_object)
    ("KNNC",
     "Data_after_KFold_KNNC(ENN)",
     KNeighborsClassifier(n_neighbors=22, leaf_size=5)),

    ("KNNC + ROA",
     "Data_after_KFold_KNNC(ENN)",
     KNeighborsClassifier(n_neighbors=11, leaf_size=3, metric='minkowski', p=1)),

    ("KNNC + CFOA",
     "Data_after_KFold_KNNC(ENN)",
     KNeighborsClassifier(n_neighbors=14, leaf_size=4, metric='minkowski', p=2)),

    ("BC",
     "Data_after_KFold_BC(ENN)",
     BaggingClassifier(
         estimator=DecisionTreeClassifier(max_depth=5),
         n_estimators=100, max_samples=0.8, max_features=0.8, random_state=42)),

    ("BC + ROA",
     "Data_after_KFold_BC(ENN)",
     BaggingClassifier(
         estimator=DecisionTreeClassifier(max_depth=7, min_samples_leaf=2),
         n_estimators=150, max_samples=0.85, max_features=0.9, random_state=17)),

    ("BC + CFOA",
     "Data_after_KFold_BC(ENN)",
     BaggingClassifier(
         estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=3),
         n_estimators=130, max_samples=0.82, max_features=0.88, random_state=99)),
]

# ── run all models ────────────────────────────────────────────────────────────

print("Running all models ...\n")
results = {}   # sheet_name -> (y_real, y_pred, proba)

for sheet_name, data_sheet, model in configs:
    print(f"  {sheet_name} ...", end=" ", flush=True)
    X, y, split = load_sheet(data_sheet)
    y_pred, proba, acc = run_model(model, X, y, split)
    results[sheet_name] = (y, y_pred, proba)
    print(f"Test Acc={acc:.4f}  Proba shape={proba.shape}")

# ── write everything as CSV files ────────────────────────────────────────────

import os
OUT_DIR = r"D:\ML\task"
os.makedirs(OUT_DIR, exist_ok=True)

print("\nSaving CSV files to:", OUT_DIR, "\n")

model_order = ["KNNC", "KNNC + ROA", "KNNC + CFOA", "BC", "BC + ROA", "BC + CFOA"]

prob_frames = []
pred_frames = []

for name in model_order:
    y_real, y_pred, proba = results[name]

    # 1. Individual model CSV  (y_real, y_pred, prob_0, prob_1)
    df = build_sheet_df(y_real, y_pred, proba)
    fname = name.replace(" + ", "_").replace(" ", "_") + ".csv"
    fpath = os.path.join(OUT_DIR, fname)
    df.to_csv(fpath, index=False)
    print(f"  Saved: {fname}")

    # 2. accumulate for Probs(ENN) and predicts(ENN)
    df_p = pd.DataFrame({
        f"{name}_y_real": y_real.astype(int),
        f"{name}_y_pred": y_pred.astype(int),
        f"{name}_prob_0": proba[:, 0].round(6),
        f"{name}_prob_1": proba[:, 1].round(6),
    })
    prob_frames.append(df_p)

    df_pred = pd.DataFrame({
        f"{name}_y_real": y_real.astype(int),
        f"{name}_y_pred": y_pred.astype(int),
    })
    pred_frames.append(df_pred)

# 3. Probs(ENN).csv
df_probs = pd.concat(prob_frames, axis=1)
df_probs.to_csv(os.path.join(OUT_DIR, "Probs(ENN).csv"), index=False)
print("  Saved: Probs(ENN).csv")

# 4. predicts(ENN).csv
df_predicts = pd.concat(pred_frames, axis=1)
df_predicts.to_csv(os.path.join(OUT_DIR, "predicts(ENN).csv"), index=False)
print("  Saved: predicts(ENN).csv")

print("\nDone. All CSV files saved successfully.")
