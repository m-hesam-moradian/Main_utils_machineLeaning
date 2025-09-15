from scipy.stats import wilcoxon, chi2_contingency
import pandas as pd


def wilcoxon_test(y1, y2):
    stat, p = wilcoxon(y1, y2)
    return stat, p


def chi_square_test(df, col1, col2, bins=5):
    crosstab = pd.crosstab(pd.qcut(df[col1], bins), pd.qcut(df[col2], bins))
    chi2, p, dof, expected = chi2_contingency(crosstab)
    return chi2, p
