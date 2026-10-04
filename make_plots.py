"""Scatter-plot matrices, PR + mirrored-ROC curves and decision trees, all driven by Data.xlsx."""
import sys, re, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import chi2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve
from xgboost import XGBClassifier

XLSX = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Data.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs/plots"
os.makedirs(OUT, exist_ok=True)
DATASETS = ["DATA", "Balanced", "RANDOM"]; SEED = 42
NO_C, YES_C = "#0099ff", "#ff00ff"
xl = pd.ExcelFile(XLSX)
frames = {d: xl.parse(d) for d in DATASETS}
LABEL = frames["DATA"].columns[-1]; FEATS = list(frames["DATA"].columns[:-1])

# ---- models/hyper-parameters from sheets ----
def clean(n):
    n = n.strip().rstrip(".,").strip(); m = re.search(r"\(([^)]+)\)", n); b = n.split("_")[0]
    return b if "_" not in n else (f"{b} {m.group(1)}" if m else n)
models = []
for sheet in ["DTC", "XGBC"]:
    raw = xl.parse(sheet, header=None); t, h, f = raw.iloc[0], raw.iloc[1], raw.iloc[2]
    keys = ("max_depth", "min_samples_split") if sheet == "DTC" else ("n_estimators", "max_depth")
    for c in t.dropna().index:
        p = {h[j]: int(f[j]) for j in range(c + 1, c + 6) if h[j] in keys}
        models.append((clean(str(t[c])), sheet, p))
def make(kind, p):
    return DecisionTreeClassifier(random_state=SEED, **p) if kind == "DTC" else \
        XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss", **p)

# ---- 1. scatter-plot matrix per dataset ----
def ellipse(ax, x, y, color, level=0.5):
    if len(x) < 5: return
    cov = np.cov(x, y); w, v = np.linalg.eigh(cov); k = np.sqrt(chi2.ppf(level, 2))
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse((x.mean(), y.mean()), 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang,
                         fill=False, ec=color, lw=1.1))
rng = np.random.RandomState(SEED)
SPLOM = [f for f in FEATS if frames['DATA'][f].nunique() > 10]   # continuous features only
n = len(SPLOM)
for ds in DATASETS:
    df = frames[ds]; sub = pd.concat([df[df[LABEL] == l].sample(min((df[LABEL] == l).sum(), 1200), random_state=SEED) for l in (0, 1)])
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), gridspec_kw=dict(wspace=0, hspace=0))
    lim = {f: (df[f].min(), df[f].max()) for f in SPLOM}
    for i, fy in enumerate(SPLOM):          # row = y
        for j, fx in enumerate(SPLOM):      # col = x
            ax = axes[i, j]
            if i == j:
                ax.text(.5, .5, fy, ha="center", va="center", fontsize=10, transform=ax.transAxes)
                ax.set_xlim(lim[fx]); ax.set_ylim(lim[fy])
            else:
                for lab, col in [(0, NO_C), (1, YES_C)]:
                    g = sub[sub[LABEL] == lab]
                    ax.scatter(g[fx], g[fy], s=9, facecolors="none", edgecolors=col, linewidths=.6, alpha=.7)
                    ellipse(ax, g[fx].values, g[fy].values, col)
                ax.set_xlim(lim[fx]); ax.set_ylim(lim[fy])
            ax.tick_params(labelsize=6.5, length=3)
            # lattice-style alternating outer axes
            top = (j % 2 == 1); right = (i % 2 == 1)
            ax.xaxis.set_visible(i == (0 if top else n - 1) and False or True)
            ax.set_xticks([]) if not ((i == 0 and j % 2 == 1) or (i == n - 1 and j % 2 == 0)) else None
            ax.set_yticks([]) if not ((j == 0 and i % 2 == 1) or (j == n - 1 and i % 2 == 0)) else None
            if i == 0 and j % 2 == 1: ax.xaxis.tick_top()
            if j == n - 1 and i % 2 == 0: ax.yaxis.tick_right()
            for s in ax.spines.values(): s.set_linewidth(.8)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=NO_C, label="No (0)"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=YES_C, label="Yes (1)")],
               loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(.5, .93))
    fig.suptitle(f"Scatter Plot Matrix – {ds}", y=.05, fontsize=13)
    fig.savefig(f"{OUT}/scatter_matrix_{ds}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("scatter", ds)

# ---- 2 + 3. fit every model on every dataset (all features, as in the sheets) ----
fits = {}
for ds in DATASETS:
    df = frames[ds]; X, y = df[FEATS].values, df[LABEL].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    for name, kind, p in models:
        clf = make(kind, p).fit(Xtr, ytr)
        fits[(name, ds)] = (clf, yte, clf.predict_proba(Xte)[:, 1])
    print("fit", ds)

def thin(a, b, k=350):
    idx = np.unique(np.linspace(0, len(a) - 1, min(k, len(a))).astype(int)); return a[idx], b[idx]
for name, kind, p in models:
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, ds in zip(axs, DATASETS):
        clf, yte, pr = fits[(name, ds)]
        prec, rec, _ = precision_recall_curve(yte, pr); fpr, tpr, _ = roc_curve(yte, pr)
        o = np.argsort(rec); rp, pp = thin(rec[o], prec[o]); rr, ff = thin(tpr, fpr)
        ax.set_facecolor("#c0c0c0"); ax.grid(axis="y", color="k", lw=.8); ax.set_axisbelow(True)
        ax.plot(rp, pp, "-s", color="#000080", ms=2.5, lw=1.5, label="Precision (PR curve)")
        ax.plot(rr, ff, "-s", color="#ff00ff", ms=2.5, lw=1.5, label="False positive rate (ROC mirror)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.2); ax.set_yticks(np.arange(0, 1.21, .2))
        ax.set_title(ds, fontsize=12, fontweight="bold"); ax.set_xlabel("Recall = True Positive Rate", fontsize=11, fontweight="bold")
    axs[0].set_ylabel("False Positive Rate (ROC Mirror); Precision (PR Curve)", fontsize=10, fontweight="bold")
    axs[0].legend(loc="lower left", fontsize=8, framealpha=.9)
    hp = ", ".join(f"{k}={v}" for k, v in p.items())
    fig.suptitle(f"Precision-Recall Graph (blue) and Mirrored ROC Curve (violet) – {name} ({hp})", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/pr_roc_{name.replace(' ', '_')}.png", dpi=120); plt.close(fig)
    print("prroc", name)

# ---- 4. decision trees (top 3 levels) for each DTC variant ----
for name, kind, p in models:
    if kind != "DTC": continue
    fig, axs = plt.subplots(1, 3, figsize=(36, 7))
    for ax, ds in zip(axs, DATASETS):
        clf = fits[(name, ds)][0]
        plot_tree(clf, max_depth=3, feature_names=FEATS, class_names=["No", "Yes"], filled=True, rounded=True,
                  impurity=True, proportion=False, fontsize=8, ax=ax)
        ax.set_title(f"{ds}  (full depth {clf.get_depth()}, {clf.get_n_leaves()} leaves; top 3 levels shown)", fontsize=13)
    hp = ", ".join(f"{k}={v}" for k, v in p.items())
    fig.suptitle(f"Decision tree based on Gini index – {name} ({hp})", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/tree_{name.replace(' ', '_')}.png", dpi=90); plt.close(fig)
    print("tree", name)
