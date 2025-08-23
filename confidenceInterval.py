import numpy as np
import pandas as pd

# داده نمونه
np.random.seed(42)
import pandas as pd

df = pd.DataFrame(
    {
        "All-Accuracy": [0.765770922, 0.998619687, 0.912306335, 0.998532155],
        "All-Precision": [0.833371635, 0.998619734, 0.912345405, 0.998532344],
        "All-Recall": [0.765770922, 0.998619687, 0.912306335, 0.998532155],
        "All-F1-score": [0.75582522, 0.998619677, 0.912316174, 0.998532136],
        "Train-Accuracy": [0.765345543, 0.999183591, 0.912383325, 0.99925934],
        "Train-Precision": [0.833252392, 0.999183594, 0.912430739, 0.999259343],
        "Train-Recall": [0.765345543, 0.999183591, 0.912383325, 0.99925934],
        "Train-F1-score": [0.755325018, 0.999183589, 0.912394396, 0.999259339],
        "Test-Accuracy": [0.767472394, 0.996364126, 0.911998384, 0.995623485],
        "Test-Precision": [0.833855362, 0.996364843, 0.912011061, 0.995627353],
        "Test-Recall": [0.767472394, 0.996364126, 0.911998384, 0.995623485],
        "Test-F1-score": [0.757822152, 0.996364032, 0.912002897, 0.995623231],
        "Value-Accuracy": [0.763735524, 0.996296795, 0.911931053, 0.995488823],
        "Value-Precision": [0.832495183, 0.996297076, 0.911956224, 0.995492206],
        "Value-Recall": [0.763735524, 0.996296795, 0.911931053, 0.995488823],
        "Value-F1-score": [0.75403643, 0.996296714, 0.911939399, 0.99548852],
        "Value-Test-Accuracy": [0.771209265, 0.996431457, 0.912065715, 0.995758147],
        "Value-Test-Precision": [0.835258858, 0.996432797, 0.912069926, 0.995762523],
        "Value-Test-Recall": [0.771209265, 0.996431457, 0.912065715, 0.995758147],
        "Value-Test-F1-score": [0.761618732, 0.996431363, 0.912067407, 0.995757945],
    }
)


def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    boot_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper


# گرفتن CI برای هر متریک
results = {}
for col in df.columns:
    results[col] = bootstrap_ci(df[col], n_bootstrap=1000, ci=95)

# نمایش نتایج
for metric, (low, high) in results.items():
    print(f"{metric}: 95% CI = ({low:.2f}, {high:.2f})")
