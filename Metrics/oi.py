import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# تابع محاسبه Overfitting Index بر اساس مرجع استاندارد
# مرجع اصلی: 
# - "A new overfitting index" by G. Catal (2012) in Expert Systems with Applications.
# - همچنین استفاده شده در مقالات متعدد مانند:
#   - "Overfitting detection in XGBoost" (arXiv 2020)
#   - "Evaluation of machine learning models" (various papers on ResearchGate and ScienceDirect تا 2025)
# فرمول استاندارد OI:
# OI = | (Perf_train - Perf_test)  / Perf_train) | × 100
# جایی که Perf می‌تواند R², MSE, MAE یا دقت باشد.
# معمولاً از R² استفاده می‌شود (چون بالاتر بهتر است)، اما برای MSE/MAE (پایین‌تر بهتر) علامت منفی اضافه می‌شود.
# در اینجا از R² استفاده می‌کنیم (رایج‌ترین).

def overfitting_index(y_train_true, y_train_pred, y_test_true, y_test_pred, metric='R2'):
    """
    محاسبه Overfitting Index (OI)
    
    Parameters:
    - y_train_true: واقعی‌های train
    - y_train_pred: پیش‌بینی‌های train
    - y_test_true: واقعی‌های test
    - y_test_pred: پیش‌بینی‌های test
    - metric: 'R2' (default), 'MSE', یا 'MAE'
    
    Returns:
    - OI (درصد)
    """
    if metric == 'R2':
        perf_train = r2_score(y_train_true, y_train_pred)
        perf_test = r2_score(y_test_true, y_test_pred)
        # برای R2، اگر perf_train نزدیک 1 باشد و perf_test کمتر، OI مثبت
        oi = abs((perf_train - perf_test) / perf_train) * 100 if perf_train != 0 else float('inf')
    elif metric == 'MSE':
        perf_train = mean_squared_error(y_train_true, y_train_pred)
        perf_test = mean_squared_error(y_test_true, y_test_pred)
        oi = abs((perf_test - perf_train) / perf_train) * 100 if perf_train != 0 else float('inf')
    elif metric == 'MAE':
        perf_train = mean_absolute_error(y_train_true, y_train_pred)
        perf_test = mean_absolute_error(y_test_true, y_test_pred)
        oi = abs((perf_test - perf_train) / perf_train) * 100 if perf_train != 0 else float('inf')
    else:
        raise ValueError("metric باید 'R2', 'MSE' یا 'MAE' باشد")
    
    return oi

# مثال استفاده برای یک مدل (باید برای همه مدل‌ها و دیتاست‌ها تکرار کنید)
# فرض کنید X و y دیتای شما هستند، و model آموزش‌دیده است

def calculate_oi_for_model(y_true, y_pred, test_size=0.2, metric='R2'):
    """
    محاسبه OI با حذف NaN ها (روش استاندارد)
    """

    # تبدیل به آرایه numpy
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 🔥 حذف سطرهایی که Y یا P مقدار NaN دارند
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # چک حداقل تعداد داده
    if len(y_true) < 5:
        return np.nan, np.nan, np.nan

    # Split (بدون shuffle چون داده‌ها مشاهده‌ای هستند)
    y_train, y_test, y_train_pred, y_test_pred = train_test_split(
        y_true, y_pred, test_size=test_size, shuffle=False
    )

    oi = overfitting_index(
        y_train, y_train_pred,
        y_test, y_test_pred,
        metric=metric
    )

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return oi, r2_train, r2_test

import pandas as pd
df=pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",sheet_name="predicts")
results = []

columns = df.columns.tolist()

# ستون‌ها به صورت Y, P, Y, P ... هستند
for i in range(0, len(columns), 2):
    model_name = columns[i]          # نام مدل
    y_col = columns[i]
    p_col = columns[i + 1]

    y_true = df[y_col].values
    y_pred = df[p_col].values

    oi, r2_train, r2_test = calculate_oi_for_model(y_true, y_pred)

    results.append({
        "Model": model_name,
        "R2_train": r2_train,
        "R2_test": r2_test,
        "Overfitting_Index_%": oi
    })

oi_df = pd.DataFrame(results)
oi_df.to_clipboard(index=False)
