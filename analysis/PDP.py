import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

# 1. خواندن داده
excel_path = r'C:\Users\Sam\Desktop\ML\task\Data.xlsx'
df = pd.read_excel(excel_path, sheet_name='RANDOM')
target_column = df.columns[-1]

# برطرف کردن اخطار FutureWarning با تبدیل داده‌ها به float
X = df.drop(columns=[target_column]).astype(float) 
y = df[target_column]

# 2. تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. آموزش مدل
model = HistGradientBoostingRegressor()
model.fit(X_train, y_train)

# 4. لیست ویژگی‌ها
features_to_plot = X.columns.tolist()

# 5. ایجاد پوشه برای ذخیره تصاویر
output_dir = r'C:\Users\Sam\Desktop\ML\task\PDP_outputs'
os.makedirs(output_dir, exist_ok=True)

# 6. رسم و ذخیره هر نمودار به صورت جداگانه
for feature in features_to_plot:
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # بررسی اینکه آیا ویژگی فقط دو مقدار دارد (مثلا صفر و یک)
    unique_values = X[feature].dropna().unique()
    is_binary = len(unique_values) <= 2
    
    if is_binary:
        # --- Custom Plotting for Binary Features (Dots instead of lines) ---
        pdp_results = partial_dependence(model, X_test, [feature])
        
        # برطرف کردن خطای KeyError با تغییر نام به grid_values
        grid_values = pdp_results['grid_values'][0] 
        average_pdp = pdp_results['average'][0]
        
        # رسم به صورت نقطه (scatter)
        ax.scatter(grid_values, average_pdp, color='blue', s=100, zorder=5)
        ax.set_xlabel(feature)
        ax.set_ylabel('Partial Dependence')
        ax.set_xticks(grid_values)
        ax.grid(True, linestyle='--', alpha=0.6)
    else:
        # --- Standard Scikit-Learn Line Plot for Continuous Features ---
        PartialDependenceDisplay.from_estimator(
            model,
            X_test,
            [feature],
            ax=ax
        )
    
    plt.tight_layout()
    plt.suptitle(f'Partial Dependence Plot - {feature}', y=1.02)

    # Naming and saving
    safe_feature_name = feature.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('%', 'pct')
    output_path = os.path.join(output_dir, f'pdp_{safe_feature_name}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

print(f'✅ همه نمودارها در پوشه "{output_dir}" ذخیره شدند.')