import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_copula_plot():
    data_file = r"D:\ML\task\Data.xlsx"
    sheet_name = "Copula"
    
    print(f"Reading {sheet_name} from {data_file}...")
    try:
        df = pd.read_excel(data_file, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error reading {data_file}: {e}")
        return

    # Check if necessary columns exist
    required_cols = ['feature_1', 'feature_2', 'sensitivity']
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing column '{col}' in the data.")
            return

    # Pivot the data to get a tabular format
    print("Pivoting data for tabular copula...")
    tabular_copula = df.pivot(index='feature_1', columns='feature_2', values='sensitivity')
    
    # Save the tabular data back to the excel file
    print("Saving tabular copula to Data.xlsx...")
    try:
        with pd.ExcelWriter(data_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            tabular_copula.to_excel(writer, sheet_name="Tabular_Copula")
    except Exception as e:
        print(f"Error saving to Excel (make sure the file is closed): {e}")
        return

    # Create the heatmap (copula plot)
    print("Generating copula plot...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(tabular_copula, annot=True, fmt=".4f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Tabular Copula Sensitivity Plot", fontsize=16)
    plt.xlabel("Feature 2 (Perturbed)", fontsize=12)
    plt.ylabel("Feature 1 (Perturbed)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    plot_path = r"D:\ML\analysis\Sensitivity\copula\copula_heatmap.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to {plot_path}")

if __name__ == "__main__":
    generate_copula_plot()
