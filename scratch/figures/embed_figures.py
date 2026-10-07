"""Embed the 4 figures into a new FIGURES sheet of the workbook."""
import os, shutil
from PIL import Image as PILImage
import openpyxl
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, Alignment

DST_XLSX  = r'd:\ML\scratch\figures\Improvement_Figures.xlsx'
PNG_DIR   = r'd:\ML\scratch\figures'
TMP_DIR   = r'd:\ML\scratch\figures\_resized'
os.makedirs(TMP_DIR, exist_ok=True)

FIGURES = [
    {'file': 'fig1_decision_boundary_matrix.png',
     'title': 'Figure 1 · Decision Boundary Matrix',
     'desc':  ('3×3 matrix. Row 1 = raw data; rows 2–3 = KNNC, RFC decision regions. '
               'Columns are dynamically chosen from top Sensitivity features. '
               'Binary target = Execution Efficiency Class (0 = Best, >0 = Other grouped).')},
    {'file': 'fig2_scatter_matrix.png',
     'title': 'Figure 2 · Per-Classifier Scatter Plot Matrix',
     'desc':  ('1×3 outer grid. One 2×2 SPLOM per classifier (KNNC, RFC) on the top-2 Sensitivity '
               'features. Each panel: main scatter + transposed scatter + 2 marginal strips, with 95% '
               'confidence ellipses per predicted class.')},
    {'file': 'fig3_binary_diagnostics.png',
     'title': 'Figure 3 · Binary Classification Diagnostics',
     'desc':  ('Four panels. (A) RFC threshold diverging bar — TP/FP/FN across 5 thresholds. '
               '(B) Information gain vs split position for Gini / Entropy / Classification error. '
               '(C) Decision tree (depth=2, Gini) on top-2 Sensitivity features. '
               '(D) Binary partitioning of predictor — three candidate splits with impurity values.')},
    {'file': 'fig4_shap_waterfall.png',
     'title': 'Figure 4 · SHAP-style Waterfall (Sensitivity)',
     'desc':  ('Side-by-side waterfalls for RFC and KNNC. Each bar = per-feature sensitivity. '
               'Cumulative red line tracks running prediction. Final bar (hatched) = total sensitivity.')},
]

TARGET_W = 1100
resized_paths = []
for fig_meta in FIGURES:
    src = os.path.join(PNG_DIR, fig_meta['file'])
    if not os.path.exists(src):
        print(f"Skipping {src} (Not found, did you run the figure script?)")
        continue
    pil = PILImage.open(src); w, h = pil.size
    if w > TARGET_W:
        pil = pil.resize((TARGET_W, int(h * TARGET_W / w)), PILImage.LANCZOS)
    dst = os.path.join(TMP_DIR, fig_meta['file'])
    pil.save(dst, 'PNG', optimize=True)
    resized_paths.append(dst)
    print(f"Resized: {fig_meta['file']} -> {pil.size}")

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'FIGURES'

TITLE_FONT   = Font(name='DejaVu Sans', size=18, bold=True, color='0F172A')
DESC_FONT    = Font(name='DejaVu Sans', size=10, italic=True, color='475569')
SECTION_FONT = Font(name='DejaVu Sans', size=13, bold=True, color='0F172A')

ws['B2'] = 'BMM-EI No.219 — Improvement Figures'; ws['B2'].font = TITLE_FONT
ws['B3'] = ('Generated from the data_after_chi2 and Morris_Sensitivity(Chi2) sheets. '
            'Binary classification target = Execution Efficiency Class (0 = Best, >0 = Other grouped).')
ws['B3'].font = DESC_FONT
ws['B3'].alignment = Alignment(wrap_text=True, vertical='top')
ws.row_dimensions[3].height = 30

for col_letter, width in [('A', 2)] + [(c, 14) for c in 'BCDEFGHIJKLMN']:
    ws.column_dimensions[col_letter].width = width

START_ROW = 5
ROW_PER_FIG = 80
for idx, (fig_meta, img_path) in enumerate(zip(FIGURES, resized_paths)):
    block_start = START_ROW + idx * ROW_PER_FIG
    cell = ws.cell(row=block_start, column=2, value=fig_meta['title']); cell.font = SECTION_FONT
    desc_cell = ws.cell(row=block_start + 1, column=2, value=fig_meta['desc'])
    desc_cell.font = DESC_FONT
    desc_cell.alignment = Alignment(wrap_text=True, vertical='top')
    ws.merge_cells(start_row=block_start + 1, start_column=2,
                    end_row=block_start + 1, end_column=14)
    ws.row_dimensions[block_start + 1].height = 32
    pil = PILImage.open(img_path); w, h = pil.size
    img = XLImage(img_path); img.width = w; img.height = h
    ws.add_image(img, f'B{block_start + 3}')
    n_rows_for_img = max(int(h / 15) + 5, 30)
    for r in range(block_start + 3, block_start + 3 + n_rows_for_img):
        ws.row_dimensions[r].height = 15

ws.sheet_properties.tabColor = '1F6FEB'
ws.freeze_panes = 'A5'
wb.save(DST_XLSX)
print(f'[OK] Saved Excel: {DST_XLSX}')
print(f'   Size: {os.path.getsize(DST_XLSX)/1024:.0f} KB')
print(f'   Sheets: {wb.sheetnames}')

wb2 = openpyxl.load_workbook(DST_XLSX)
ws2 = wb2['FIGURES']
print(f'\nVerification: {len(ws2._images)} images in FIGURES sheet')
for i, img in enumerate(ws2._images):
    print(f'  Image {i+1}: {img.width}x{img.height}')
