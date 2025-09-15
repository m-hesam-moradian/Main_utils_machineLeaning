import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(
    df: pd.DataFrame,
    target: str,
    excel_path: str = "data/data.xlsx",
    sheet_name: str = "VIF_Results",
    save_to_excel: bool = False,
) -> pd.DataFrame:
    df = df.dropna()
    features = list(df.drop(target, axis=1).columns)
    all_tables = []

    while True:
        print("✅ Current features:", features)

        X = pd.get_dummies(df[features], drop_first=True)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            X = X.drop(columns=non_numeric)

        X_with_const = np.column_stack((np.ones(X.shape[0]), X.values))
        columns_with_const = ["const"] + list(X.columns)
        X_df = pd.DataFrame(X_with_const, columns=columns_with_const)

        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_df.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])
        ]

        all_tables.append(vif_data)  # no iteration column

        vif_no_const = vif_data[vif_data["Variable"] != "const"]
        max_vif = vif_no_const["VIF"].max()
        highest_vif_var = vif_no_const.loc[vif_no_const["VIF"].idxmax(), "Variable"]

        print(f"🚨 Highest VIF:", highest_vif_var, "=", max_vif)

        if max_vif < 1:
            break

        if highest_vif_var in features:
            features.remove(highest_vif_var)
        else:
            for f in features:
                if highest_vif_var.startswith(f + "_") or highest_vif_var == f:
                    features.remove(f)
                    break

    # Pad tables to same number of rows
    max_rows = max(len(tbl) for tbl in all_tables)
    padded_tables = []
    for tbl in all_tables:
        if len(tbl) < max_rows:
            pad = pd.DataFrame([["", ""]] * (max_rows - len(tbl)), columns=tbl.columns)
            tbl = pd.concat([tbl, pad], ignore_index=True)
        padded_tables.append(tbl)

    # Merge side by side with 1 column space
    # Merge side by side with empty columns as spacer
    final_df = padded_tables[0]
    for i, tbl in enumerate(padded_tables[1:], start=1):
        # Create 1 empty column spacer
        spacer = pd.DataFrame([[""] * 2] * max_rows, columns=["", ""])
        final_df = pd.concat([final_df, spacer, tbl], axis=1)

    if save_to_excel:
        try:
            with pd.ExcelWriter(
                excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
            ) as writer:
                final_df.to_excel(writer, sheet_name=sheet_name, index=False)
        except FileNotFoundError:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                final_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"📁 VIF results saved to '{excel_path}' in sheet '{sheet_name}'")
    return final_df
