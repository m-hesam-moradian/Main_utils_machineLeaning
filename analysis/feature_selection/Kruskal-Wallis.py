import pandas as pd
from scipy.stats import kruskal


def kruskal_selection(df, target_col="Vehicle Speed", q_groups=3, alpha=0.05):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Create groups of target
    df["Target_Group"] = pd.qcut(y, q=q_groups, labels=False)

    selected_features = []
    for col in X.columns:
        try:
            groups = [
                X[df["Target_Group"] == g][col] for g in df["Target_Group"].unique()
            ]
            stat, p_value = kruskal(*groups)
            if p_value < alpha:
                selected_features.append(col)
            print(f"{col}: p={p_value:.4f}")
        except Exception as e:
            print(f"Skipping {col}: {e}")

    return selected_features


if __name__ == "__main__":
    file_path = "Vehicle-Specific and Traffic _dataset.xlsx"
    df = pd.read_excel(file_path, sheet_name="sim_11").dropna()
    features = kruskal_selection(df, "Vehicle Speed", q_groups=3)
    print("✅ Kruskal Selected Features:", features)
