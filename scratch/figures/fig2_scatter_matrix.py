"""
Figure 2: Per-classifier scatter plot matrix with 95% confidence ellipses.
5 mini-2x2 SPLOMs (one per classifier) + 1 legend panel.
"""
import sys, os
sys.path.insert(0, r'd:\ML\task\codes\novelty figures')
from fig_common import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

X, y, feats = load_data()
copula = load_copula()
top_features_rfc = get_top_features(copula, 'RFC', k=2)
fx, fy = top_features_rfc[0], top_features_rfc[1]
print(f"Top features for scatter: {fx} vs {fy}")

all_cls = train_classifiers(X, y, feature_subset=feats)

FEATURE_DISPLAY = {f: f.replace('_', ' ') for f in feats}
XL = FEATURE_DISPLAY.get(fx, fx)
YL = FEATURE_DISPLAY.get(fy, fy)

OUT_ROWS, OUT_COLS = 1, 3
INNER = 2
fig = plt.figure(figsize=(18, 6.5))
outer_gs = gridspec.GridSpec(OUT_ROWS, OUT_COLS, figure=fig,
                              wspace=0.30, hspace=0.40,
                              left=0.06, right=0.97, top=0.92, bottom=0.05)

# fig.suptitle('Scatter Plot Matrix per Binary Classifier  ·  Top-2 Sensitivity Features',
#              x=0.06, y=0.965, ha='left', fontsize=16, fontweight='bold', color=SLATE)
# fig.text(0.06, 0.935,
#          f'Features: {XL}  vs  {YL}    ·    '
#          f'Target: Execution Efficiency (0=Best, 1=Other)    ·    '
#          f'Ellipses: 95% confidence regions per class',
#          ha='left', fontsize=10, style='italic', color=SLATE_2)

for idx, (clf_name, info) in enumerate(all_cls.items()):
    r, c = divmod(idx, OUT_COLS)
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        INNER, INNER, subplot_spec=outer_gs[r, c],
        wspace=0.10, hspace=0.10,
        width_ratios=[2, 1], height_ratios=[1, 2])
    X_te, y_te, y_pred, acc = info['X_test'], info['y_test'], info['y_pred'], info['acc']
    x_idx, y_idx = feats.index(fx), feats.index(fy)
    xs = X_te.iloc[:, x_idx].values
    ys = X_te.iloc[:, y_idx].values

    ax_main  = fig.add_subplot(sub_gs[0, 0])
    ax_margx = fig.add_subplot(sub_gs[0, 1], sharey=ax_main)
    ax_margy = fig.add_subplot(sub_gs[1, 0], sharex=ax_main)
    ax_trans = fig.add_subplot(sub_gs[1, 1], sharex=ax_margx, sharey=ax_margy)

    mask_p0, mask_p1 = (y_pred == 0), (y_pred == 1)
    rng = np.random.RandomState(42)
    n_show = 600
    idx0 = rng.choice(np.where(mask_p0)[0], size=min(n_show, mask_p0.sum()), replace=False)
    idx1 = rng.choice(np.where(mask_p1)[0], size=min(n_show, mask_p1.sum()), replace=False)
    idx_all = np.concatenate([idx0, idx1])

    ax_main.scatter(xs[idx0], ys[idx0], c=COLOR_NORMAL, s=12, alpha=0.55,
                    edgecolors='white', linewidth=0.3, zorder=3)
    ax_main.scatter(xs[idx1], ys[idx1], c=COLOR_ATTACK, s=12, alpha=0.65,
                    edgecolors='white', linewidth=0.3, zorder=3)
    if len(idx0) >= 3:
        confidence_ellipse(xs[idx0], ys[idx0], ax_main,
                            facecolor=COLOR_NORMAL, edgecolor=COLOR_NORMAL, alpha=0.30, n_std=2.0)
    if len(idx1) >= 3:
        confidence_ellipse(xs[idx1], ys[idx1], ax_main,
                            facecolor=COLOR_ATTACK, edgecolor=COLOR_ATTACK, alpha=0.30, n_std=2.0)

    ax_trans.scatter(ys[idx0], xs[idx0], c=COLOR_NORMAL, s=12, alpha=0.55,
                     edgecolors='white', linewidth=0.3, zorder=3)
    ax_trans.scatter(ys[idx1], xs[idx1], c=COLOR_ATTACK, s=12, alpha=0.65,
                     edgecolors='white', linewidth=0.3, zorder=3)
    if len(idx0) >= 3:
        confidence_ellipse(ys[idx0], xs[idx0], ax_trans,
                            facecolor=COLOR_NORMAL, edgecolor=COLOR_NORMAL, alpha=0.30, n_std=2.0)
    if len(idx1) >= 3:
        confidence_ellipse(ys[idx1], xs[idx1], ax_trans,
                            facecolor=COLOR_ATTACK, edgecolor=COLOR_ATTACK, alpha=0.30, n_std=2.0)

    ax_margx.scatter(np.zeros_like(xs[idx_all]) + np.random.RandomState(1).uniform(-0.15, 0.15, len(idx_all)),
                     xs[idx_all],
                     c=[COLOR_NORMAL if p==0 else COLOR_ATTACK for p in y_pred[idx_all]],
                     s=8, alpha=0.5, edgecolors='none', zorder=3)
    ax_margx.set_xticks([]); ax_margx.set_xlim(xs.min(), xs.max())

    ax_margy.scatter(ys[idx_all],
                     np.zeros_like(ys[idx_all]) + np.random.RandomState(2).uniform(-0.15, 0.15, len(idx_all)),
                     c=[COLOR_NORMAL if p==0 else COLOR_ATTACK for p in y_pred[idx_all]],
                     s=8, alpha=0.5, edgecolors='none', zorder=3)
    ax_margy.set_yticks([]); ax_margy.set_ylim(ys.min(), ys.max())

    ax_main.set_title(f'{clf_name}', fontsize=13, fontweight='bold',
                      color=CLASSIFIER_PALETTE[clf_name], pad=6, loc='left')
    ax_main.text(0.98, 0.04, f'Acc = {acc:.3f}',
                 transform=ax_main.transAxes, ha='right', va='bottom',
                 fontsize=10, fontweight='bold', color=INK,
                 bbox=dict(boxstyle='round,pad=0.30', facecolor='white',
                           edgecolor=BG_3, linewidth=0.6, alpha=0.92))
    ax_margy.set_xlabel(XL, fontsize=9, color=SLATE_2)
    ax_margy.set_ylabel(YL, fontsize=9, color=SLATE_2, rotation=0, ha='right', va='center')
    ax_margy.yaxis.set_label_coords(-0.005, 0.5)

    for a in (ax_margx, ax_trans):
        a.tick_params(axis='both', labelsize=8, colors=SLATE_2)
    ax_main.tick_params(axis='both', labelsize=8, colors=SLATE_2)
    ax_margy.tick_params(axis='both', labelsize=8, colors=SLATE_2)
    for a in (ax_main, ax_margx, ax_trans):
        a.set_xlabel(''); a.set_ylabel('')
    for a in (ax_main, ax_trans, ax_margx, ax_margy):
        a.grid(True, alpha=0.15, linestyle='--', color=BG_3, linewidth=0.5)
        a.set_axisbelow(True)

ax_leg = fig.add_subplot(outer_gs[0, OUT_COLS - 1])
ax_leg.set_axis_off()
ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
ax_leg.text(0.05, 0.95, 'Legend & Notes', fontsize=12, fontweight='bold', color=SLATE, ha='left', va='top')
ax_leg.scatter([0.10], [0.82], c=COLOR_NORMAL, s=80, edgecolor='white', linewidth=1)
ax_leg.text(0.18, 0.82, 'Class 0', fontsize=10, color=INK, va='center')
ax_leg.scatter([0.10], [0.74], c=COLOR_ATTACK, s=80, edgecolor='white', linewidth=1)
ax_leg.text(0.18, 0.74, 'Class 1/2', fontsize=10, color=INK, va='center')
ell_normal = Ellipse((0.10, 0.66), width=0.06, height=0.04,
                     facecolor=COLOR_NORMAL, alpha=0.25, edgecolor=COLOR_NORMAL, lw=1.2)
ell_attack = Ellipse((0.10, 0.60), width=0.06, height=0.04,
                     facecolor=COLOR_ATTACK, alpha=0.25, edgecolor=COLOR_ATTACK, lw=1.2)
ax_leg.add_patch(ell_normal); ax_leg.add_patch(ell_attack)
ax_leg.text(0.18, 0.63, '95% confidence ellipse per class', fontsize=10, color=INK, va='center')

notes = [
    'Layout per panel:',
    '   top-left:      main scatter (y vs x) by predicted class',
    '   top-right:     marginal strip of feature x',
    '   bottom-left:   marginal strip of feature y',
    '   bottom-right:  transposed scatter (x vs y)',
    '',
    f'Features selected: top-2 by Sensitivity (RFC)',
    f'   · {XL}', f'   · {YL}', '',
    'Classifiers: KNNC, RFC',
    'All trained on the full RFE selected feature set.',
]
y_pos = 0.50
for line in notes:
    ax_leg.text(0.05, y_pos, line, fontsize=9, color=SLATE_2, ha='left', va='top', family='monospace')
    y_pos -= 0.034

out = os.path.join(DOWNLOAD_DIR, 'fig2_scatter_matrix.png')
fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
print(f'[OK] Saved: {out}')
