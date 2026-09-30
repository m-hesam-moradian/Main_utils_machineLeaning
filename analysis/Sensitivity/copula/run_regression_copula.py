import pandas as pd
import numpy as np
import seaborn as plt_sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor

def couples_sensitivity_analysis(
    model, X, y, feature_pairs, metric="mse", perturbation=0.1
):
    if metric == "mse":
        metric_func = mean_squared_error
    elif metric == "mae":
        metric_func = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    elif metric == "accuracy":
        metric_func = accuracy_score
    else:
        raise ValueError("Unsupported metric")

    original_predictions = model.predict(X)
    original_score = metric_func(y, original_predictions)

    sensitivity_report = []

    for feature_1, feature_2 in feature_pairs:
        X_perturbed = X.copy()
        
        # Apply perturbation
        if isinstance(X_perturbed, pd.DataFrame):
            X_perturbed[feature_1] *= 1 + perturbation
            X_perturbed[feature_2] *= 1 + perturbation
        else:
            X_perturbed[:, feature_1] *= 1 + perturbation
            X_perturbed[:, feature_2] *= 1 + perturbation

        perturbed_predictions = model.predict(X_perturbed)
        perturbed_score = metric_func(y, perturbed_predictions)
        sensitivity = perturbed_score - original_score

        sensitivity_report.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "original_score": original_score,
                "perturbed_score": perturbed_score,
                "sensitivity": sensitivity,
            }
        )

    return pd.DataFrame(sensitivity_report)


def run_full_copula():
    data_file = r"D:\ML\task\Data.xlsx"
    sheet_name = "Data_after_KFold_LSSVR(MRMR)"
    
    print(f"Reading {sheet_name} from {data_file}...")
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    features = X.columns
    
    # Train model
    print("Training XGBRegressor...")
    model = XGBRegressor(random_state=42)
    model.fit(X, y)
    
    feature_pairs = [
        (features[i], features[j])
        for i in range(len(features))
        for j in range(len(features))
    ]
    
    print("Running sensitivity analysis (regression, metric='mse', perturbation=0.1)...")
    copula = couples_sensitivity_analysis(
        model, X, y, feature_pairs, metric="mse", perturbation=0.1
    )
    
    print("Pivoting data for tabular copula...")
    tabular_copula = copula.pivot(index='feature_1', columns='feature_2', values='sensitivity')
    
    print("Saving Copula and Tabular_Copula to Excel...")
    with pd.ExcelWriter(data_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        copula.to_excel(writer, sheet_name="Copula", index=False)
        tabular_copula.to_excel(writer, sheet_name="Tabular_Copula")
        
    print("Generating copula heatmap...")
    plt.figure(figsize=(12, 10))
    plt_sns.heatmap(tabular_copula, annot=True, fmt=".4f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Tabular Copula Sensitivity Plot (Regression - MSE)", fontsize=16)
    plt.xlabel("Feature 2 (Perturbed by 10%)", fontsize=12)
    plt.ylabel("Feature 1 (Perturbed by 10%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plot_path = r"D:\ML\analysis\Sensitivity\copula\copula_heatmap.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to {plot_path}")

if __name__ == "__main__":
    run_full_copula()
