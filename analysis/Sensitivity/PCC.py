import pandas as pd

datasets = [r'C:\Users\Sam\Desktop\ML\task\Data.xlsx']


corr_results = {}

for file in datasets:
    # خواندن فایل
    df = pd.read_excel(file,sheet_name='Data_after_KFold_ADAC(IQR)')
    target = df.columns[-1]
    # فقط ستون‌های عددی
    numeric_df = df.select_dtypes(include='number')

    # محاسبه Pearson Correlation با متغیر هدف
    corr_df = (
        numeric_df.corr(method='pearson')[[target]]
        .sort_values(by=target, ascending=False)
        .reset_index()
    )

    corr_df.columns = ['Feature', 'Pearson_Correlation']

    corr_results[file] = corr_df

    print(f"\n{'='*50}")
    print(f"Dataset: {file}")
    print(corr_df)
    corr_df.to_clipboard()

# دسترسی به دیتافریم‌ها
df_s1_corr = corr_results[r'C:\Users\Sam\Desktop\ML\task\Data.xlsx']
