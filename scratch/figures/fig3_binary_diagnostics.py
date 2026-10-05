"""
Figure 3: Binary Classification Diagnostics — 4 panels
(A) RFC threshold diverging bar (TP/FP/FN) 
(B) Information gain curves for Gini/Entropy/Classification error
(C) Decision tree (depth=2, Gini) on top-2 Copula features
(D) Binary partitioning bands with 3 candidate splits + impurity values
"""
import sys, os
sys.path.insert(0, r'd:\ML\task\codes\novelty figures')
from fig_common import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

X, y, feats = load_data()
copula = load_copula()
top_features = get_top_features(copula, 'RFC', k=2)
fx, fy = top_features[0], top_features[1]

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, figure=fig,
                       width_ratios=[1.0, 1.0],
                       height_ratios=[1.05, 0.95, 1.10],
                       wspace=0.22, hspace=0.45,
                       left=0.06, right=0.97, top=0.91, bottom=0.04)

# fig.suptitle('Binary Classification Diagnostics',
#              x=0.06, y=0.97, ha='left', fontsize=17, fontweight='bold', color=SLATE)
# fig.text(0.06, 0.94,
#          'Target: Execution Efficiency (0=Best, 1=Other)    ·    '
#          'Top Sensitivity features used for partitions and tree',
#          ha='left', fontsize=10, style='italic', color=SLATE_2)

# ═══ Panel A: Threshold diverging bar (spans 2 cols) ═══
ax_thr = fig.add_subplot(gs[0, :])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X[feats].values, y, test_size=0.25, random_state=42, stratify=y)
rfc = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=2)
rfc.fit(X_train, y_train)
proba = rfc.predict_proba(X_test)[:, 1]

thresholds = [0.25, 0.45, 0.55, 0.75, 0.90]
TP, FP, FN = [], [], []
for t in thresholds:
    y_pred = (proba >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    TP.append(tp); FP.append(fp); FN.append(-fn)

x_pos = np.arange(len(thresholds))
ax_thr.bar(x_pos, TP, width=0.55, color='#5DADE2', edgecolor='white',
           linewidth=0.7, zorder=3, label='TP  (True Positives)')
ax_thr.bar(x_pos, FP, bottom=TP, width=0.55, color='#7F8C8D', edgecolor='white',
           linewidth=0.7, zorder=3, label='FP  (False Positives)')
ax_thr.bar(x_pos, FN, width=0.55, color='#E0533D', edgecolor='white',
           linewidth=0.7, zorder=3, label='FN  (False Negatives)')
ax_thr.axhline(0, color=SLATE, linewidth=1.2, zorder=4)
for i, (tp, fp, fn) in enumerate(zip(TP, FP, FN)):
    ax_thr.text(i, tp + fp + 12, f'{tp + fp}', ha='center', va='bottom',
                fontsize=9, color=SLATE, fontweight='bold')
    ax_thr.text(i, fn - 12, f'{-fn}', ha='center', va='top',
                fontsize=9, color=SLATE, fontweight='bold')
ax_thr.set_xticks(x_pos)
ax_thr.set_xticklabels([f'τ = {t:.2f}' for t in thresholds], fontsize=10, color=INK)
ax_thr.set_xlabel('Decision Threshold  τ', fontsize=10, color=SLATE_2, labelpad=6)
ax_thr.set_ylabel('Number of Classified Items', fontsize=10, color=SLATE_2)
ax_thr.set_title('(A) Threshold-Based Diverging Bar — RFC Classifier',
                 loc='left', fontsize=12, fontweight='bold', color=SLATE, pad=4)
ax_thr.legend(loc='upper right', bbox_to_anchor=(1.00, 1.00), fontsize=9, frameon=False, ncol=3)
ax_thr.grid(True, axis='y', alpha=0.18, linestyle='--', color=BG_3, linewidth=0.5)
ax_thr.set_axisbelow(True)
ax_thr.set_ylim(min(FN) - 30, max(t + f for t, f in zip(TP, FP)) + 50)

# ═══ Panel B: Information gain curves ═══
ax_ig = fig.add_subplot(gs[1, 0])
feature_for_split = fx
x_vals = X[feature_for_split].values
y_vals = y

def gini(y_arr):
    if len(y_arr) == 0: return 0
    p = np.bincount(y_arr, minlength=2) / len(y_arr)
    return 1.0 - np.sum(p ** 2)

def entropy(y_arr):
    if len(y_arr) == 0: return 0
    p = np.bincount(y_arr, minlength=2) / len(y_arr)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def clf_error(y_arr):
    if len(y_arr) == 0: return 0
    p = np.bincount(y_arr, minlength=2) / len(y_arr)
    return 1.0 - np.max(p)

parent_gini = gini(y_vals)
parent_entropy = entropy(y_vals)
parent_error = clf_error(y_vals)

n_splits = 60
qs = np.linspace(0.05, 0.95, n_splits)
split_points = np.quantile(x_vals, qs)
ig_gini, ig_entropy, ig_error = [], [], []
for sp in split_points:
    mask_left = x_vals <= sp; mask_right = ~mask_left
    n_l, n_r = mask_left.sum(), mask_right.sum()
    n_total = len(x_vals)
    if n_l == 0 or n_r == 0:
        ig_gini.append(0); ig_entropy.append(0); ig_error.append(0); continue
    gl, gr = gini(y_vals[mask_left]), gini(y_vals[mask_right])
    el, er = entropy(y_vals[mask_left]), entropy(y_vals[mask_right])
    cl, cr = clf_error(y_vals[mask_left]), clf_error(y_vals[mask_right])
    ig_gini.append(parent_gini - (n_l/n_total) * gl - (n_r/n_total) * gr)
    ig_entropy.append(parent_entropy - (n_l/n_total) * el - (n_r/n_total) * er)
    ig_error.append(parent_error - (n_l/n_total) * cl - (n_r/n_total) * cr)

ax_ig.plot(split_points, ig_entropy, color='#1F6FEB', linewidth=2.2,
           label='$IG_e$  (Entropy)', zorder=3)
ax_ig.plot(split_points, ig_error, color='#B26B00', linewidth=2.2, linestyle='--',
           label='$IG_c$  (Classification error)', zorder=3)
ax_ig.plot(split_points, ig_gini, color='#5B237A', linewidth=2.2, linestyle=':',
           label='$IG_g$  (Gini)', zorder=3)
best_idx = int(np.argmax(ig_gini))
ax_ig.axvline(split_points[best_idx], color=SLATE_2, linestyle='--',
               linewidth=1.0, alpha=0.6, zorder=2)
ax_ig.text(split_points[best_idx], max(ig_entropy) * 1.02,
           f' best Gini split\n  x = {split_points[best_idx]:.2f}',
           fontsize=9, color=SLATE, ha='left', va='bottom')
ax_ig.set_xlabel(feature_for_split.replace('_', ' ').title(), fontsize=10, color=SLATE_2)
ax_ig.set_ylabel('Information Gain', fontsize=10, color=SLATE_2)
ax_ig.set_title('(B) Partitioning Information Gain — Three Impurity Metrics',
                loc='left', fontsize=12, fontweight='bold', color=SLATE, pad=4)
ax_ig.legend(loc='upper right', bbox_to_anchor=(1.00, 1.00), fontsize=9, frameon=False, ncol=1)
ax_ig.grid(True, alpha=0.18, linestyle='--', color=BG_3, linewidth=0.5)
ax_ig.set_axisbelow(True)
ax_ig.set_ylim(0, max(max(ig_entropy), max(ig_gini), max(ig_error)) * 1.15)

# ═══ Panel C: Decision tree (depth=2) ═══
ax_tree = fig.add_subplot(gs[1, 1])
dtc = DecisionTreeClassifier(max_depth=2, criterion='gini', random_state=42)
dtc.fit(X[[fx, fy]].values, y)
plot_tree(dtc, feature_names=[fx, fy], class_names=['Class 0', 'Class 1/2'],
          filled=True, rounded=True, fontsize=10, ax=ax_tree,
          impurity=True, proportion=True, label='root')
ax_tree.set_title('(C) Decision Tree (Gini, depth = 2) — Top-2 Features',
                  loc='left', fontsize=12, fontweight='bold', color=SLATE, pad=4)

# ═══ Panel D: Binary partitioning bands ═══
ax_part = fig.add_subplot(gs[2, :])
splits_to_show = [split_points[max(int(n_splits*0.20), best_idx-3)],
                  split_points[best_idx],
                  split_points[min(int(n_splits*0.85), best_idx+5)]]
y_pos = np.arange(len(splits_to_show))
for i, sp in enumerate(splits_to_show):
    mask_left = x_vals <= sp; mask_right = ~mask_left
    n_l, n_r = mask_left.sum(), mask_right.sum()
    n_total = n_l + n_r
    h_l, h_r = n_l / n_total, n_r / n_total
    c0_l = (y_vals[mask_left] == 0).sum(); c1_l = (y_vals[mask_left] == 1).sum()
    c0_r = (y_vals[mask_right] == 0).sum(); c1_r = (y_vals[mask_right] == 1).sum()
    y_l = y_pos[i] - 0.32; y_r = y_pos[i] + 0.08
    p_l0 = c0_l / n_l if n_l else 0
    p_l1 = c1_l / n_l if n_l else 0
    ax_part.barh(y_l, p_l0 * h_l, height=0.36, color='#5B8DEF', edgecolor='white', linewidth=0.6)
    ax_part.barh(y_l, p_l1 * h_l, left=p_l0 * h_l, height=0.36, color='#E0533D', edgecolor='white', linewidth=0.6)
    p_r0 = c0_r / n_r if n_r else 0
    p_r1 = c1_r / n_r if n_r else 0
    ax_part.barh(y_r, p_r0 * h_r, height=0.36, color='#5B8DEF', edgecolor='white', linewidth=0.6)
    ax_part.barh(y_r, p_r1 * h_r, left=p_r0 * h_r, height=0.36, color='#E0533D', edgecolor='white', linewidth=0.6)
    ax_part.text(p_l0 * h_l / 2, y_l + 0.18,
                 f'$I_g$={gini(y_vals[mask_left]):.2f}\nn₁={n_l}',
                 ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax_part.text(p_l0 * h_l + p_l1 * h_l / 2, y_l + 0.18, f'{c1_l}',
                 ha='center', va='center', fontsize=8, color='white')
    ax_part.text(p_l0 * h_l + (p_l1 * h_l) / 2, y_l - 0.13,
                 f'class-1 share = {p_l1:.2f}', ha='center', va='center', fontsize=7.5, color=SLATE_2)
    ax_part.text(h_l + p_r0 * h_r / 2, y_r + 0.18,
                 f'$I_g$={gini(y_vals[mask_right]):.2f}\nn₂={n_r}',
                 ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax_part.text(h_l + p_r0 * h_r + p_r1 * h_r / 2, y_r + 0.18, f'{c1_r}',
                 ha='center', va='center', fontsize=8, color='white')
    ax_part.text(h_l + p_r0 * h_r + (p_r1 * h_r) / 2, y_r - 0.13,
                 f'class-1 share = {p_r1:.2f}', ha='center', va='center', fontsize=7.5, color=SLATE_2)
    ax_part.axvline(h_l, color=SLATE, linestyle=':', linewidth=1.0, alpha=0.7, zorder=4)
    ax_part.text(h_l + 0.005, y_pos[i] + 0.30, f'x = {sp:.2f}',
                 fontsize=9, color=SLATE, ha='left', va='bottom', fontweight='bold')

ax_part.text(0.005, y_pos[0] - 0.42,
             f'Parent Gini  $I_g$ = {parent_gini:.3f}   ·   Total samples  N = {len(x_vals)}',
             fontsize=9, color=SLATE_2, ha='left', va='top')
ax_part.set_yticks(y_pos)
ax_part.set_yticklabels([f'Split #{i+1}' for i in range(len(splits_to_show))], fontsize=10, color=INK)
ax_part.set_xlabel(f'Normalised partition size (left ⊕ right) on  {feature_for_split.replace("_", " ").title()}',
                   fontsize=10, color=SLATE_2)
ax_part.set_xlim(0, 1.05)
ax_part.set_title('(D) Binary Partitioning of Predictor — Three Candidate Splits',
                  loc='left', fontsize=12, fontweight='bold', color=SLATE, pad=4)
ax_part.grid(True, axis='x', alpha=0.18, linestyle='--', color=BG_3, linewidth=0.5)
ax_part.set_axisbelow(True)
ax_part.invert_yaxis()
legend_elems = [Patch(facecolor='#5B8DEF', label='Class 0'),
                Patch(facecolor='#E0533D', label='Class 1/2')]
ax_part.legend(handles=legend_elems, loc='upper right',
               bbox_to_anchor=(1.00, 1.00), fontsize=9, frameon=False, ncol=2)

out = os.path.join(DOWNLOAD_DIR, 'fig3_binary_diagnostics.png')
fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
print(f'[OK] Saved: {out}')
