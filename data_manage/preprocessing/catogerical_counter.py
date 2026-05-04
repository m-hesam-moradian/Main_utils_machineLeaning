import pandas as pd

def categorical_summary_to_clipboard(excel_path, unique_threshold=10):
    # Load data
    df = pd.read_excel(excel_path)

    categorical_cols = []

    # detect categorical features
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        dtype = df[col].dtype

        if dtype == "object" or dtype == "bool" or nunique <= unique_threshold:
            categorical_cols.append(col)

    # build table
    rows = []

    for col in categorical_cols:
        value_counts = df[col].value_counts(dropna=False)

        for val, count in value_counts.items():
            rows.append([col, val, count])

    summary_df = pd.DataFrame(rows, columns=["Feature", "Value", "Count"])

    # copy to clipboard
    summary_df.to_clipboard(index=False)

    print("📋 Categorical summary table copied to clipboard!")

# Example usage
categorical_summary_to_clipboard(
    excel_path=r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
)