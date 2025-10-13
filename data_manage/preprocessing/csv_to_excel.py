import pandas as pd

# خواندن فایل CSV
df = pd.read_csv(
    r"D:\ML\Main_utils\task\EL. No 6. Allocated bandwidth- SVR-ENR-SCO-POA-GGO-DATA.csv"
)

# ذخیره به صورت فایل اکسل
df.to_excel(
    r"D:\ML\Main_utils\task\EL. No 6. Allocated bandwidth- SVR-ENR-SCO-POA-GGO-DATA.xlsx",
    sheet_name="DATA"
    index=False,
)

print("✅ فایل dataset_invade.xlsx با موفقیت ساخته شد.")
