import openpyxl

wb = openpyxl.load_workbook('task/Data.xlsx', data_only=True)
ws = wb['LGBC(No_SMOTE) + GOA']

print('Merged ranges in LGBC(No_SMOTE) + GOA:', [str(m) for m in ws.merged_cells.ranges])
for c in range(1, 35):
    val = ws.cell(row=2, column=c).value
    if val is not None:
        print(f'  Col {c} ({openpyxl.utils.get_column_letter(c)}): "{val}"')

wb.close()
