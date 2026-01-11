import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Sheet2"

# خواندن داده‌ها از اکسل
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# محاسبه ماتریس همبستگی
corr_matrix = df.corr()

# نمایش ماتریس همبستگی به صورت گرافیکی
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("ماتریس همبستگی ویژگی‌ها و هدف")
plt.show()