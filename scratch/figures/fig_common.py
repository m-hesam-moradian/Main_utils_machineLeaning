"""
Shared data loader and style setup for BMM-EI No.221 figure generation.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse, Patch
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family':     'sans-serif',
    'axes.unicode_minus': False,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor':   '#FFFFFF',
    'axes.edgecolor':   '#475569',
    'axes.linewidth':   0.9,
    'axes.spines.top':   False,
    'axes.spines.right':  False,
    'axes.grid':         False,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.color':       '#475569',
    'ytick.color':       '#475569',
    'xtick.labelsize':  9, 'ytick.labelsize': 9,
    'axes.labelsize':   10, 'axes.labelcolor':  '#1F2937',
    'axes.titlesize':   12, 'axes.titleweight': 'bold',
    'axes.titlecolor':  '#0F172A', 'axes.titlepad': 8,
    'legend.frameon':  False, 'legend.fontsize': 9,
    'figure.dpi':        110, 'savefig.dpi':       300,
    'savefig.bbox':     'tight', 'savefig.facecolor': '#FFFFFF',
    'savefig.pad_inches': 0.25,
})

SLATE   = '#0F172A'
INK     = '#1F2937'
SLATE_2 = '#475569'
MUTED   = '#94A3B8'
BG_1    = '#F8FAFC'
BG_2    = '#EEF2F7'
BG_3    = '#E2E8F0'
COLOR_NORMAL  = '#5B8DEF'
COLOR_ATTACK  = '#E0533D'
COLOR_NEUTRAL = '#94A3B8'
CLASSIFIER_PALETTE = {
    'MLR': '#1F6FEB', 'QDA':  '#0E7C66',
    'KNNC': '#1F6FEB', 'RFC':  '#5B237A',
}
REGION_CMAP = LinearSegmentedColormap.from_list('binary_region', [(0, '#E8F0FE'), (1, '#FDE4DD')])

EXCEL_PATH   = r'd:\ML\task\Data.xlsx'
DOWNLOAD_DIR = r'd:\ML\scratch\figures'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def load_data():
    df = pd.read_excel(EXCEL_PATH, sheet_name='data_after_vif', engine='openpyxl')
    target_col = 'Routing Decision'
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y_raw = df[target_col].astype(int).values
    # Convert to binary for the plot scripts (Class 0 vs Class 1/2) if required
    y = (y_raw > 0).astype(int) 
    return X, y, feature_cols

def load_copula():
    # Use CAM Sensitivity instead of Copula/Morris
    df = pd.read_excel(EXCEL_PATH, sheet_name='CAM_Sensitivity(Unbalanced)', engine='openpyxl')
    out = {'MLR': [], 'QDA': []}
    for _, row in df.iterrows():
        if pd.notna(row['Feature']):
            out['MLR'].append((row['Feature'], float(row['MLR(Unbalanced)'])))
            out['QDA'].append((row['Feature'], float(row['QDA(Unbalanced)'])))
    return out

def get_top_features(copula_dict, model='MLR', k=2, exclude_categorical=True):
    items = sorted(copula_dict[model], key=lambda x: x[1], reverse=True)
    return [f for f, _ in items[:k]]

def train_classifiers(X, y, feature_subset=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    X_train = X if feature_subset is None else X[feature_subset]
    classifiers = {
        'MLR': LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000),
        'QDA': QuadraticDiscriminantAnalysis(reg_param=0.0, tol=0.0001, store_covariance=False),
    }
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y, test_size=0.25, random_state=42, stratify=y)
    out = {}
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        out[name] = {'clf': clf, 'acc': accuracy_score(y_te, y_pred), 'y_pred': y_pred,
                     'X_train': X_tr, 'X_test': X_te, 'y_train': y_tr, 'y_test': y_te}
    return out

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if len(x) < 3: return None
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height, angle=angle,
                  facecolor=facecolor, edgecolor=kwargs.get('edgecolor', SLATE),
                  lw=1.4, alpha=kwargs.get('alpha', 0.9), zorder=4)
    ax.add_patch(ell)
    return ell
