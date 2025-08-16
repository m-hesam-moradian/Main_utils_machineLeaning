import numpy as np


def generate_fake_predictions(y_true, desired_accuracy, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    y_true = np.array(y_true)
    n_samples = len(y_true)
    n_correct = int(desired_accuracy * n_samples)
    n_wrong = n_samples - n_correct

    # پیدا کردن برچسب‌های ممکن
    unique_labels = np.unique(y_true)

    # ایندکس‌های تصادفی درست و غلط
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    correct_indices = indices[:n_correct]
    wrong_indices = indices[n_correct:]

    # ساخت y_pred
    y_pred = np.empty_like(y_true)

    # درست‌ها: همان برچسب واقعی
    y_pred[correct_indices] = y_true[correct_indices]

    # غلط‌ها: انتخاب یک برچسب متفاوت از برچسب واقعی
    for idx in wrong_indices:
        wrong_choices = unique_labels[unique_labels != y_true[idx]]
        y_pred[idx] = np.random.choice(wrong_choices)

    return y_pred


y = np.loadtxt(r"D:\ML\Main_utils\class_noice\y.txt")

desired_accuracy = 0.9

y_pred = generate_fake_predictions(y, desired_accuracy, random_seed=42)

print = y_pred
