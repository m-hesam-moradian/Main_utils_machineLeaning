import pandas as pd

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

# Select sheets
keywords = ["MLR", "KNN"]

all_sheets = pd.ExcelFile(excel_path).sheet_names

if keywords:
    sheet_names = [
        s for s in all_sheets
        if any(k.lower() in s.lower() for k in keywords)
    ]
else:
    sheet_names = all_sheets

print("Matching sheets:")
print(sheet_names)


merged_columns = []

for sheet in sheet_names:

    # Read sheet including headers
    df = pd.read_excel(
        excel_path,
        sheet_name=sheet,
        header=2
    )

    # -----------------------------------------
    # Find prediction + probability columns
    # -----------------------------------------

    cols = []

    for col in df.columns:

        col_name = str(col).lower()

        if (
            "y_real" in col_name
            or "y_pred" in col_name
            or "prob" in col_name
        ):
            cols.append(col)


    # Keep only required columns
    df = df[cols]


    # If columns have no names (original layout)
    if len(df.columns) == 0:

        raw = pd.read_excel(
            excel_path,
            sheet_name=sheet,
            header=None,
            skiprows=2
        )

        # first two + probability columns only
        df = raw.iloc[:, :6]


    # -----------------------------------------
    # Rename dynamically
    # -----------------------------------------

    new_cols = []

    for i in range(df.shape[1]):

        if i == 0:
            new_cols.append(f"{sheet}_y_real")

        elif i == 1:
            new_cols.append(f"{sheet}_y_pred")

        else:
            new_cols.append(
                f"{sheet}_prob_{i-2}"
            )

    df.columns = new_cols

    merged_columns.append(df)


# Merge
df_merged = pd.concat(
    merged_columns,
    axis=1
)


# Copy
df_merged.to_clipboard(index=False)

print("\nOnly prediction + probability columns copied.")
print(df_merged.head())
print("\nShape:", df_merged.shape)