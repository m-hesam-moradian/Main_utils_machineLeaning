"""
Figure 1: Decision Boundary Matrix
6 rows (raw data + 5 classifiers) × 3 columns (feature pairs)
"""
import sys, os
sys.path.insert(0, r'd:\ML\task\codes\novelty figures')
from fig_common import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.tree       import DecisionTreeClassifier
from sklearn.ensemble   import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
HAS_XGB = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y, feats = load_data()
FEATURE_DISPLAY = {f: f.replace('_', ' ') for f in feats}
PAIRS = [
    (feats[0], feats[1]),
    (feats[2], feats[3]),
    (feats[4], feats[5])
]
CLASSIFIERS = {
    'KNNC': lambda: KNeighborsClassifier(n_neighbors=7),
    'RFC':  lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=2),
}

N_ROWS = 1 + len(CLASSIFIERS)
N_COLS = len(PAIRS)
fig = plt.figure(figsize=(15, 9), constrained_layout=False)
gs = gridspec.GridSpec(N_ROWS, N_COLS, figure=fig,
                       wspace=0.18, hspace=0.32,
                       left=0.10, right=0.97, top=0.89, bottom=0.04)

# fig.suptitle('Decision Boundaries — How Binary Classifiers Partition the Feature Space',
#              x=0.045, y=0.985, ha='left', fontsize=17, fontweight='bold', color=SLATE)
# fig.text(0.045, 0.955,
#          'Binary target: Execution Efficiency Class (0 = Best, >0 = Other classes grouped)',
#          ha='left', fontsize=10, style='italic', color=SLATE_2)
# for c, (fx, fy) in enumerate(PAIRS):
#     fig.text((c + 0.5) / N_COLS * 0.87 + 0.10, 0.925,
#             #  f'Pair {c+1}:  {FEATURE_DISPLAY.get(fx, fx)}  vs  {FEATURE_DISPLAY.get(fy, fy)}',
#              ha='center', fontsize=11, fontweight='bold', color=INK)

REGION_CMAP = ListedColormap(['#DCE7FA', '#FAD7CE'])

for r_idx, (clf_name, clf_factory) in enumerate([(None, None)] + list(CLASSIFIERS.items())):
    for c_idx, (fx, fy) in enumerate(PAIRS):
        ax = fig.add_subplot(gs[r_idx, c_idx])
        X_pair = X[[fx, fy]].values
        x_min, x_max = X_pair[:, 0].min(), X_pair[:, 0].max()
        y_min, y_max = X_pair[:, 1].min(), X_pair[:, 1].max()
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        if r_idx == 0:
            mask0, mask1 = y == 0, y == 1
            ax.scatter(X_pair[mask0, 0], X_pair[mask0, 1],
                       c=COLOR_NORMAL, s=8, alpha=0.45, edgecolors='none', zorder=3)
            ax.scatter(X_pair[mask1, 0], X_pair[mask1, 1],
                       c=COLOR_ATTACK,  s=8, alpha=0.55, edgecolors='none', zorder=3)
            ax.set_title('Raw Data', fontsize=11, fontweight='bold', color=INK, pad=4)
        else:
            clf = clf_factory()
            X_tr, X_te, y_tr, y_te = train_test_split(X_pair, y, test_size=0.25,
                                                      random_state=42, stratify=y)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc = accuracy_score(y_te, y_pred)

            n_grid = 100
            xx, yy = np.meshgrid(np.linspace(x_min - x_pad, x_max + x_pad, n_grid),
                                  np.linspace(y_min - y_pad, y_max + y_pad, n_grid))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=REGION_CMAP, levels=[-0.5, 0.5, 1.5], alpha=0.55, zorder=1)
            n_show = min(400, len(X_te))
            idx = np.random.RandomState(42).choice(len(X_te), n_show, replace=False)
            Xs, ys, yp = X_te[idx], y_te[idx], y_pred[idx]
            correct = ys == yp
            ax.scatter(Xs[correct & (ys==0), 0], Xs[correct & (ys==0), 1],
                       c=COLOR_NORMAL, s=14, alpha=0.85, edgecolors='white', linewidth=0.4, zorder=3)
            ax.scatter(Xs[correct & (ys==1), 0], Xs[correct & (ys==1), 1],
                       c=COLOR_ATTACK,  s=14, alpha=0.85, edgecolors='white', linewidth=0.4, zorder=3)
            if (~correct).any():
                ax.scatter(Xs[~correct, 0], Xs[~correct, 1],
                           c='#1E293B', s=18, alpha=0.7, marker='x', linewidths=0.9, zorder=5)
            ax.set_title(f'{clf_name}', fontsize=11, fontweight='bold',
                         color=CLASSIFIER_PALETTE[clf_name], pad=4)
            ax.text(0.97, 0.04, f'Acc = {acc:.3f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9, fontweight='bold', color=INK,
                    bbox=dict(boxstyle='round,pad=0.30', facecolor='white',
                              edgecolor=BG_3, linewidth=0.6, alpha=0.92))

        ax.set_xlabel(FEATURE_DISPLAY.get(fx, fx) if r_idx == N_ROWS-1 else '',
                      fontsize=9, color=SLATE_2)
        ax.set_ylabel(FEATURE_DISPLAY.get(fy, fy) if c_idx == 0 else '',
                      fontsize=9, color=SLATE_2)
        ax.grid(True, alpha=0.18, linestyle='--', color=BG_3, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=8, colors=SLATE_2)

legend_elems = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_NORMAL,
               markersize=8, label='Class 0'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ATTACK,
               markersize=8, label='Class 1/2'),
    plt.Line2D([0], [0], marker='x', color='#1E293B', markersize=7,
               markeredgewidth=1.0, label='Misclassified'),
]
fig.legend(handles=legend_elems, loc='upper right',
           bbox_to_anchor=(0.97, 0.96), fontsize=9, frameon=False, ncol=1)

out = os.path.join(DOWNLOAD_DIR, 'fig1_decision_boundary_matrix.png')
fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
print(f'[OK] Saved: {out}')
