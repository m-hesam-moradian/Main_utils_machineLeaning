import pandas as pd
import numpy as np
import os
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier

def main():
    filepath = r"task\Data.xlsx"

    xl = pd.ExcelFile(filepath)
    if "data_after_chi2" in xl.sheet_names:
        sheet_name = "data_after_chi2"
    elif "data_after_vif" in xl.sheet_names:
        sheet_name = "data_after_vif"
    elif "Selected_Data_RFE" in xl.sheet_names:
        sheet_name = "Selected_Data_RFE"
    elif "Z-Score" in xl.sheet_names:
        sheet_name = "Z-Score"
    elif "DATA_Shuffled" in xl.sheet_names:
        sheet_name = "DATA_Shuffled"
    else:
        sheet_name = "Data"

    print(f"Reading dataset for SHAP from sheet: '{sheet_name}'")
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    target_column = df.columns[-1]
    X_raw = df.drop(columns=[target_column]).values
    y = df[target_column].values
    feature_names = list(df.drop(columns=[target_column]).columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(X_scaled, y))

    model_factories = {
        "LGBC": lambda f: LGBMClassifier(
            n_estimators=20,
            learning_rate=0.00001,
            random_state=42 + f,
            n_jobs=-1,
            verbose=-1
        ),
        "ETC": lambda f: ExtraTreesClassifier(
            n_estimators=6,
            max_depth=3,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features=0.4,
            random_state=42 + f,
            n_jobs=-1
        ),
        "BC": lambda f: ExtraTreesClassifier(
            n_estimators=5,
            max_depth=4,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features=0.4,
            random_state=42 + f,
            n_jobs=-1
        )
    }

    shap_reports = {}

    for model_name, factory in model_factories.items():
        print(f"\nEvaluating SHAP for 5-Fold Cross Validation: {model_name}...")
        shap_records = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            m = factory(fold_idx)
            m.fit(X_tr, y_tr)

            print(f"  -> Fold {fold_idx}: Calculating SHAP values...")
            try:
                explainer = shap.TreeExplainer(m)
                shap_values = explainer(X_te)
                shap_array = shap_values.values
                if len(shap_array.shape) == 3:
                    shap_importance = np.abs(shap_array).mean(axis=(0, 2))
                else:
                    shap_importance = np.abs(shap_array).mean(axis=0)
                
                for f_name, f_val in zip(feature_names, shap_importance):
                    shap_records.append({"Fold": fold_idx, "Feature": f_name, "SHAP_Importance": f_val})
            except Exception as e:
                print(f"  -> SHAP failed for Fold {fold_idx}: {e}")

        if shap_records:
            df_shap = pd.DataFrame(shap_records)
            shap_reports[model_name] = df_shap
            print(f"\nSHAP Summary for {model_name}:")
            shap_summary = df_shap.groupby("Feature")["SHAP_Importance"].mean().reset_index().sort_values(by="SHAP_Importance", ascending=False)
            print(shap_summary.to_string(index=False))

    shap_report_path = r"D:\ML\SHAP_KFold_Report.xlsx"
    try:
        with pd.ExcelWriter(shap_report_path, engine="openpyxl") as writer:
            for model_name, df_shap in shap_reports.items():
                df_shap.to_excel(writer, sheet_name=f"{model_name}_SHAP", index=False)
                shap_summary = df_shap.groupby("Feature")["SHAP_Importance"].mean().reset_index().sort_values(by="SHAP_Importance", ascending=False)
                shap_summary.to_excel(writer, sheet_name=f"{model_name}_SHAP_Mean", index=False)
        print(f"\n[+] SHAP report successfully saved to {shap_report_path}")
    except Exception as e:
        print(f"Could not save SHAP report: {e}")

if __name__ == "__main__":
    main()
