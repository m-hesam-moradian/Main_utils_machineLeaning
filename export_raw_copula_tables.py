import sys, os
sys.path.append(r'F:\ML\analysis\Sensitivity\copula')

import pandas as pd
import numpy as np
import openpyxl
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

excel_path = r'task/BMM-EI. No.09-Results.xlsx'

print("Loading workbook...", flush=True)
wb = openpyxl.load_workbook(excel_path, data_only=True)

def sheet_to_df(sheet_name):
    ws = wb[sheet_name]
    data = ws.values
    cols = next(data)
    return pd.DataFrame(data, columns=cols).dropna(how='all')

df_bal = sheet_to_df('BALANC')
df_dat = sheet_to_df('DATA')

features_no_ids = [c for c in df_bal.columns if c not in ['IDS_Alerts', 'Anomalous_Load']]
features_with_ids = [c for c in df_dat.columns if c != 'Anomalous_Load']

pairs_bal = [(features_no_ids[i], features_no_ids[j]) for i in range(len(features_no_ids)) for j in range(len(features_no_ids))]
pairs_dat = [(features_with_ids[i], features_with_ids[j]) for i in range(len(features_with_ids)) for j in range(len(features_with_ids))]

# ------------------------------------------------------------------
# Model Training (per classCopula.py: ExtraTreesClassifier)
# ------------------------------------------------------------------
model_bal = ExtraTreesClassifier(max_depth=7, n_estimators=15, random_state=42)
model_bal.fit(df_bal[features_no_ids].astype(float), df_bal['Anomalous_Load'].astype(int))

model_dat = ExtraTreesClassifier(max_depth=7, n_estimators=15, random_state=42)
model_dat.fit(df_dat[features_with_ids].astype(float), df_dat['Anomalous_Load'].astype(int))

model_dat_no = ExtraTreesClassifier(max_depth=7, n_estimators=15, random_state=42)
model_dat_no.fit(df_dat[features_no_ids].astype(float), df_dat['Anomalous_Load'].astype(int))

# ------------------------------------------------------------------
# Continuous Class-Conditional Couples Sensitivity Function
# ------------------------------------------------------------------
def compute_class_couples_sensitivity(model, X_subset, target_class, feature_pairs, pert=0.1):
    orig_prob = model.predict_proba(X_subset)[:, target_class].mean()
    records = []
    for f1, f2 in feature_pairs:
        X_p = X_subset.copy()
        X_p[f1] *= 1 + pert
        X_p[f2] *= 1 + pert
        new_prob = model.predict_proba(X_p)[:, target_class].mean()
        sens = new_prob - orig_prob
        records.append({'feature_1': f1, 'feature_2': f2, 'sensitivity': sens})
    df_records = pd.DataFrame(records)
    tabular = df_records.pivot(index='feature_1', columns='feature_2', values='sensitivity')
    return tabular, df_records

# 1. BALANC Class 0 (without IDS_Alerts)
X_bal_0 = df_bal[df_bal['Anomalous_Load'] == 0][features_no_ids].astype(float)
tab_bal_0, rep_bal_0 = compute_class_couples_sensitivity(model_bal, X_bal_0, target_class=0, feature_pairs=pairs_bal, pert=0.1)

# 2. BALANC Class 1 (without IDS_Alerts)
X_bal_1 = df_bal[df_bal['Anomalous_Load'] == 1][features_no_ids].astype(float)
tab_bal_1, rep_bal_1 = compute_class_couples_sensitivity(model_bal, X_bal_1, target_class=1, feature_pairs=pairs_bal, pert=0.1)

# 3. DATA Class 0 (with all features)
X_dat_0 = df_dat[df_dat['Anomalous_Load'] == 0][features_with_ids].astype(float)
tab_dat_0, rep_dat_0 = compute_class_couples_sensitivity(model_dat, X_dat_0, target_class=0, feature_pairs=pairs_dat, pert=0.1)

# 4. DATA Class 1 (with all features)
X_dat_1 = df_dat[df_dat['Anomalous_Load'] == 1][features_with_ids].astype(float)
tab_dat_1, rep_dat_1 = compute_class_couples_sensitivity(model_dat, X_dat_1, target_class=1, feature_pairs=pairs_dat, pert=0.1)

# Verify zero counts
print("Verification of non-zero sensitivity:")
print("  tab_bal_0 zeros:", (tab_bal_0 == 0).sum().sum(), "out of", tab_bal_0.size)
print("  tab_bal_1 zeros:", (tab_bal_1 == 0).sum().sum(), "out of", tab_bal_1.size)
print("  tab_dat_0 zeros:", (tab_dat_0 == 0).sum().sum(), "out of", tab_dat_0.size)
print("  tab_dat_1 zeros:", (tab_dat_1 == 0).sum().sum(), "out of", tab_dat_1.size)

# ------------------------------------------------------------------
# Ablation Summary DataFrame
# ------------------------------------------------------------------
y_dat = df_dat['Anomalous_Load'].astype(int)
acc_with = model_dat.score(df_dat[features_with_ids].astype(float), y_dat)
acc_without = model_dat_no.score(df_dat[features_no_ids].astype(float), y_dat)

mi_all = dict(zip(features_with_ids, mutual_info_classif(df_dat[features_with_ids].astype(float), y_dat, random_state=42)))
mi_no = dict(zip(features_no_ids, mutual_info_classif(df_dat[features_no_ids].astype(float), y_dat, random_state=42)))

avg_dat_all = rep_dat_0.groupby('feature_1')['sensitivity'].mean()

# Compute sensitivity without IDS
X_dat_all_no = df_dat[features_no_ids].astype(float)
orig_prob_no = model_dat_no.predict_proba(X_dat_all_no)[:, 0].mean()
recs_no = []
for f1, f2 in pairs_bal:
    X_p = X_dat_all_no.copy()
    X_p[f1] *= 1.10
    X_p[f2] *= 1.10
    s = model_dat_no.predict_proba(X_p)[:, 0].mean() - orig_prob_no
    recs_no.append({'feature_1': f1, 'feature_2': f2, 'sensitivity': s})
avg_dat_no = pd.DataFrame(recs_no).groupby('feature_1')['sensitivity'].mean()

ablation_rows = []
for f in features_with_ids:
    s_w = float(avg_dat_all.loc[f])
    if f in avg_dat_no:
        s_wo = float(avg_dat_no.loc[f])
        shift = s_wo - s_w
        mi_w = float(mi_all.get(f, 0.0))
        mi_wo = float(mi_no.get(f, 0.0))
        diag = "Redistributed Sensitivity Post-Ablation" if abs(s_wo) > abs(s_w) else "Retained Secondary Feature"
    else:
        s_wo = np.nan
        shift = np.nan
        mi_w = float(mi_all.get(f, 0.0))
        mi_wo = np.nan
        diag = f"Target Leakage Ablated (Acc Drop: -{(acc_with - acc_without)*100:.2f}%)"
    
    ablation_rows.append({
        'Feature': f,
        'Avg_Sensitivity_With_IDS': s_w,
        'Avg_Sensitivity_Without_IDS': s_wo,
        'Sensitivity_Shift': shift,
        'Mutual_Info_With_IDS': mi_w,
        'Mutual_Info_Without_IDS': mi_wo,
        'Ablation_Diagnostic': diag
    })

df_ablation = pd.DataFrame(ablation_rows)

# ------------------------------------------------------------------
# Write RAW tables using pandas ExcelWriter (exact script style)
# ------------------------------------------------------------------
print(f"Writing raw tables to {excel_path}...", flush=True)
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    tab_bal_0.to_excel(writer, sheet_name="COPULA_BALANC_Class0")
    tab_bal_1.to_excel(writer, sheet_name="COPULA_BALANC_Class1")
    tab_dat_0.to_excel(writer, sheet_name="COPULA_DATA_Class0")
    tab_dat_1.to_excel(writer, sheet_name="COPULA_DATA_Class1")
    df_ablation.to_excel(writer, sheet_name="Ablation_Study", index=False)

print("All 4 raw copula tables and ablation study exported successfully!", flush=True)

