"""Decision-boundary matrix built entirely from Data.xlsx.
Columns = datasets (sheets DATA, Balanced, RANDOM)
Rows    = models (DTC / XGBC and their optimizer variants, hyper-parameters read from the DTC & XGBC sheets)
Plot features = the 2 features with the highest total-order index (ST) in the FAST sheet.
"""
import sys, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

XLSX = sys.argv[1] if len(sys.argv) > 1 else r"d:\ML\task\Data.xlsx"
OUT  = sys.argv[2] if len(sys.argv) > 2 else r"d:\ML\task\decision_boundaries_matrix.png"
DATASETS = ["DATA", "Balanced", "RANDOM"]
N_SHOW, SEED = 160, 42
xl = pd.ExcelFile(XLSX)

# ---- features from FAST sheet (highest ST) ----
fast = xl.parse("FAST", header=None).dropna(how="all").dropna(how="all", axis=1)
hdr = fast.index[fast.eq("column_name").any(axis=1)][0]
fast.columns = fast.loc[hdr]; fast = fast.loc[hdr + 1:]
fast["ST"] = fast["ST"].astype(float)
FEATS = fast.sort_values("ST", ascending=False)["column_name"].head(2).tolist()

# ---- models + hyper-parameters from DTC / XGBC sheets ----
def clean(name):
    name = name.strip().rstrip(".,").strip()
    m = re.search(r"\(([^)]+)\)", name)
    base = name.split("_")[0]
    return base if "_" not in name else f"{base}\n{m.group(1)}" if m else name
models = []   # (label, kind, params)
for sheet in ["DTC", "XGBC"]:
    raw = xl.parse(sheet, header=None)
    titles = raw.iloc[0]; head = raw.iloc[1]; first = raw.iloc[2]
    for c in titles.dropna().index:
        label = clean(str(titles[c]))
        hp_cols = [j for j in range(c + 1, c + 6) if isinstance(head[j], str) and head[j] not in ("P_M", "P_P") and head[j] != "Number" and not pd.isna(head[j]) and j < c + 6 and head[j] in
                   (("max_depth", "min_samples_split") if sheet == "DTC" else ("n_estimators", "max_depth"))]
        params = {head[j]: int(first[j]) for j in hp_cols}
        models.append((label, sheet, params))

def make(kind, p):
    if kind == "DTC":
        return DecisionTreeClassifier(random_state=SEED, **p)
    return XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss", **p)

# ---- figure ----
cm_bg = plt.cm.RdBu            # red (class 0) -> blue (class 1)
cm_pt = ListedColormap(["#c0111f", "#0a1fbf"])
nr, nc = len(models) + 1, len(DATASETS)
fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc + 2.2, 2.55 * nr))
rng = np.random.RandomState(SEED)
for j, ds in enumerate(DATASETS):
    df = xl.parse(ds)
    X = df[FEATS].values; y = df["Fraud Label"].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    itr = rng.choice(len(Xtr_s), N_SHOW, replace=False); ite = rng.choice(len(Xte_s), N_SHOW // 2, replace=False)
    pad = 0.5
    lo, hi = Xtr_s.min(0) - pad, Xtr_s.max(0) + pad
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 220), np.linspace(lo[1], hi[1], 220))
    grid = np.c_[xx.ravel(), yy.ravel()]
    def points(ax):
        ax.scatter(*Xtr_s[itr].T, c=ytr[itr], cmap=cm_pt, edgecolors="k", s=16, linewidths=.5)
        ax.scatter(*Xte_s[ite].T, c=yte[ite], cmap=cm_pt, edgecolors="k", s=16, linewidths=.5, alpha=.6)
        ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max()); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[0, j]; points(ax); ax.set_title(ds, fontsize=18, fontweight="bold")
    for i, (label, kind, p) in enumerate(models, start=1):
        clf = make(kind, p).fit(Xtr_s, ytr)
        acc = clf.score(Xte_s, yte)
        Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax = axes[i, j]
        ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap=cm_bg, alpha=.8, vmin=0, vmax=1)
        points(ax)
        ax.text(xx.max() - .05 * (xx.max() - xx.min()), yy.min() + .05 * (yy.max() - yy.min()),
                f"{acc:.2f}".lstrip("0") if acc < 1 else "1.0", ha="right", va="bottom", fontsize=16, fontweight="bold")
        if j == 0:
            ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=20,
                          fontsize=16, fontweight="bold")
axes[0, 0].set_ylabel(f"Features:\n{FEATS[0]}\nvs\n{FEATS[1]}", rotation=0, ha="right", va="center",
                      labelpad=20, fontsize=12, color="#444", fontweight="bold")
fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="#c0111f", mec="#c0111f", label="Class 0"),
                    Line2D([], [], marker="o", ls="", mfc="#0a1fbf", mec="#0a1fbf", label="Class 1"),
                    Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="Train (solid)"),
                    Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="Test (faded)")],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=14, ncol=4)
fig.tight_layout(rect=(0.06, 0, 1, 0.95))
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("features:", FEATS); print("saved", OUT)
