import pandas as pd
from openpyxl import load_workbook


def clean_missing_samples(excel_path, source_sheet="DATA", target_sheet="CLEANED_DATA"):
    # Load the data from the source sheet
    df = pd.read_excel(excel_path, sheet_name=source_sheet)

    # Drop rows with any missing values
    cleaned_df = df.dropna()

    # Load the workbook to check existing sheets
    book = load_workbook(excel_path)
    writer_args = dict(engine="openpyxl", mode="a")

    # If the target sheet already exists, remove it first
    if target_sheet in book.sheetnames:
        del book[target_sheet]
        book.save(excel_path)

    # Write the cleaned data to the new sheet
    # with pd.ExcelWriter(excel_path, **writer_args) as writer:
    #     cleaned_df.to_excel(writer, sheet_name=target_sheet, index=False)
    cleaned_df.to_clipboard(index=False)
    print(f"🧹 Cleaned data saved to sheet '{target_sheet}' in '{excel_path}'.")


# Example usage
clean_missing_samples(
    excel_path=r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    source_sheet="Encoded_Data",
    target_sheet="CLEANED_DATA",
)
