import pandas as pd

# خواندن فایل CSV با جداکننده ;
df = pd.read_csv(
    r"C:\Users\Sam\Downloads\scarcity.csv",
    sep=";"
)

# ذخیره به صورت فایل اکسل
df.to_excel(
    r"C:\Users\Sam\Downloads\scarcity.xlsx",
    sheet_name="DATA",
    index=False,
)

print("✅ فایل Excel با موفقیت ساخته شد.")