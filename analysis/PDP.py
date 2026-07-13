import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

# 1. خواندن داده
excel_path = r'C:\Users\Sam\Desktop\ML\task\Data.xlsx'
df = pd.read_excel(excel_path, sheet_name='Data_after_KFold_QR(ANOVA)')
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# 2. تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. آموزش مدل
model = HistGradientBoostingRegressor(
    # max_depth=20,          # Maximum depth of each tree
    # random_state=42       # Random seed for reproducibility
)
model.fit(X_train, y_train)

# 4. لیست ویژگی‌ها (در صورت نیاز تغییر بده)
features_to_plot = X.columns.tolist()  # همه ویژگی‌ها

# 5. ایجاد پوشه برای ذخیره تصاویر
output_dir = r'C:\Users\Sam\Desktop\ML\task\PDP_outputs'
os.makedirs(output_dir, exist_ok=True)

# 6. رسم و ذخیره هر نمودار به صورت جداگانه
for feature in features_to_plot:
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        model,
        X_test,
        [feature],
        ax=ax
    )
    plt.tight_layout()
    plt.suptitle(f'Partial Dependence Plot - {feature}', y=1.02)

    safe_feature_name = feature.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('%', 'pct')
    output_path = os.path.join(output_dir, f'pdp_{safe_feature_name}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

print(f'✅ همه نمودارها در پوشه "{output_dir}" ذخیره شدند.')
