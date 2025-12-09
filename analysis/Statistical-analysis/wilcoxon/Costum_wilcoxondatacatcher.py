import pandas as pd

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"

def extract_and_concat_sheets_horizontal(file_path):
    sheet_names = pd.ExcelFile(file_path).sheet_names
    print("Sheets found:", sheet_names)

    sheet_dfs = []

    for sheet_name in sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")

        # Read first row for titles
        df_header = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=1)
        if df_header.empty:
            print(f"⚠ Skipping sheet '{sheet_name}' (empty sheet).")
            continue

        first_row_strings = [str(x).strip() for x in df_header.iloc[0] if isinstance(x, str)]
        if not first_row_strings:
            print(f"⚠ Skipping sheet '{sheet_name}' (no string titles found).")
            continue

        # Read actual data from second row
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        if df.empty:
            print(f"⚠ Skipping sheet '{sheet_name}' (no data).")
            continue

        df.columns = df.columns.map(lambda x: str(x).strip())

        # Detect y/p columns
        # y_real	y_pred

        y_cols = [c for c in df.columns if c.lower().startswith("y")]
        p_cols = [c for c in df.columns if c.lower().startswith("p")]

        if not y_cols or not p_cols:
            print(f"⚠ Skipping sheet '{sheet_name}' (no Y/P columns found).")
            continue

        # Sort numerically
        def sort_key(x):
            try:
                return int(x.split(".")[1])
            except:
                return -1
        y_cols = sorted(y_cols, key=sort_key)
        p_cols = sorted(p_cols, key=sort_key)

        # Concatenate Y/P pairs horizontally
        pairs = [df[[y, p]] for y, p in zip(y_cols, p_cols)]
        df_final = pd.concat(pairs, axis=1)

        # Rename columns using first row strings
        for i, title in enumerate(first_row_strings):
            col_y_idx = i*2
            col_p_idx = i*2 + 1
            if col_y_idx < len(df_final.columns):
                df_final.columns.values[col_y_idx] = f"{title}"
            if col_p_idx < len(df_final.columns):
                df_final.columns.values[col_p_idx] = f"{title}"

        sheet_dfs.append(df_final)

    # Concatenate all sheets horizontally by aligning on index
    if sheet_dfs:
        final_df = pd.concat(sheet_dfs, axis=1)
        final_df.to_clipboard(index=False)
        print("\n✔ DONE! Concatenated horizontally and copied to clipboard.")
        print(final_df.head())
        return final_df
    else:
        print("❌ No valid sheets with Y/P columns.")
        return None

# Run the function
extract_and_concat_sheets_horizontal(excel_path)
