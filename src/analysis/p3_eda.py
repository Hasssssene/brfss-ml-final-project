"""P3 EDA — report figures for the 76-column matrix v1 (clean_v1.parquet).

Produces (figures/, PNG dpi=150):
  1. eda_target_imbalance.png        — raw GENHLTH classes + binary target
  2. eda_missingness_matrix.png      — missing % per column (76 bars)
  3. eda_association_top25.png       — top-25 |Spearman rho| vs binary target
  4. eda_correlation_heatmap.png     — Spearman matrix of top-40 by |rho|
  5. eda_key_features_by_class.png   — class-conditional distributions

Side artifacts (results/analysis/):
  eda_missingness.csv, eda_association_spearman.csv

Throwaway analysis snippet (data-analyst). Only matplotlib/seaborn/pandas/numpy.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects
from matplotlib.patches import Rectangle

from src.config import DATA_INTERIM, FIGURES, RESULTS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- shared style ----
C_NEG = "#4C72B0"   # y = 0 / negative rho
C_POS = "#DD8452"   # y = 1 / positive rho
C_TOP = "#C44E52"   # highlight (top-10 missingness, block outlines)
C_REST = "#9AA5B1"  # neutral bars
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})
ANALYSIS = RESULTS / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Load + valid mask
# =====================================================================
X = pd.read_parquet(DATA_INTERIM / "clean_v1.parquet")
T = pd.read_parquet(DATA_INTERIM / "target_v1.parquet")
valid = T["valid"].to_numpy()
Xv = X.loc[valid].reset_index(drop=True)
y = T.loc[valid, "y"].to_numpy()
raw = T.loc[valid, "genhlth_raw"].to_numpy()
n = len(y)
print(f"matrix {X.shape} -> valid rows {n}, positives {y.sum()} ({y.mean()*100:.4f}%)")

GENHLTH_LBL = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

# =====================================================================
# Figure 1 — target imbalance
# =====================================================================
raw_counts = pd.Series(raw).value_counts().sort_index()
n_dropped = int((~valid).sum())

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)

ax = axes[0]
codes = [1, 2, 3, 4, 5]
cnts = [int((raw == c).sum()) for c in codes]
cols = [C_NEG, C_NEG, C_NEG, C_POS, C_POS]
bars = ax.bar(range(5), cnts, color=cols, edgecolor="white", width=0.72)
for i, (b, c) in enumerate(zip(bars, cnts)):
    ax.text(b.get_x() + b.get_width() / 2, c + 2500,
            f"{c:,}\n({c/n*100:.1f}%)", ha="center", va="bottom", fontsize=10)
ax.set_xticks(range(5))
ax.set_xticklabels([f"{c}\n{GENHLTH_LBL[c]}" for c in codes])
ax.set_ylabel("Respondents")
ax.set_ylim(0, max(cnts) * 1.18)
ax.set_title("(a) Raw GENHLTH, 5 classes")
ax.text(0.985, 0.97, f"valid rows n = {n:,}\n{n_dropped:,} rows (DK/Refused/blank) excluded",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F2F2F2", ec="#CCCCCC"))
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=C_NEG), plt.Rectangle((0, 0), 1, 1, fc=C_POS)],
          labels=["y = 0  (classes 1–3)", "y = 1  (classes 4–5)"],
          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=10)

ax = axes[1]
n0, n1 = int((y == 0).sum()), int((y == 1).sum())
bars = ax.bar([0, 1], [n0, n1], color=[C_NEG, C_POS], edgecolor="white", width=0.55)
for b, c in zip(bars, [n0, n1]):
    ax.text(b.get_x() + b.get_width() / 2, c + 6000, f"{c:,}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(b.get_x() + b.get_width() / 2, c / 2, f"{c/n*100:.2f} %",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")
ax.set_xticks([0, 1])
ax.set_xticklabels(["y = 0\nGood or better\n(GENHLTH 1–3)", "y = 1\nFair / Poor\n(GENHLTH 4–5)"])
ax.set_ylabel("Respondents")
ax.set_ylim(0, n0 * 1.2)
ax.set_title("(b) Binary target: 1 : 5.0 class imbalance")

fig.suptitle("Target definition and class imbalance — BRFSS 2021, GENHLTH recoded to binary",
             fontsize=13.5, fontweight="bold")
fig.savefig(FIGURES / "eda_target_imbalance.png")
plt.close(fig)
print("fig 1 done")

# =====================================================================
# Figure 2 — missingness per column
# =====================================================================
miss = (Xv.isna().mean() * 100).sort_values(ascending=False)
miss.to_frame("missing_pct").rename_axis("column").to_csv(ANALYSIS / "eda_missingness.csv")
n50, n25 = int((miss > 50).sum()), int((miss > 25).sum())

fig, ax = plt.subplots(figsize=(9.2, 13.2))
labels = miss.index.tolist()
vals = miss.to_numpy()
colors_ = [C_TOP if i < 10 else C_REST for i in range(len(vals))]
ax.barh(range(len(vals)), vals, color=colors_, edgecolor="white", height=0.72)
ax.invert_yaxis()
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Missing values after sentinel recode (%)")
ax.set_xlim(0, 88)
for i in range(10):
    ax.text(vals[i] + 0.7, i + 0.02, f"{vals[i]:.1f}", va="center", fontsize=8, color=C_TOP)
ax.axvline(25, color="#555555", ls="--", lw=1.1)
ax.axvline(50, color="#555555", ls="--", lw=1.1)
ax.text(25.6, len(vals) - 1.2, "25 %", color="#555555", fontsize=9, rotation=0)
ax.text(50.6, len(vals) - 1.2, "50 %", color="#555555", fontsize=9)
ax.set_title("Missingness of the 76-column matrix v1 (valid rows, sorted)")
ax.text(0.985, 0.03,
        f"n = {n:,} valid rows • 19 columns > 50 % • 21 columns > 25 %\n"
        f"median 5.3 % • 4 columns complete (SEXVAR, QSTLANG, _STATE, _AGE80)\n"
        f"top-10 highlighted (crimson): optional-module + conditional items",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="#FDF3F3", ec=C_TOP, lw=0.8))
fig.savefig(FIGURES / "eda_missingness_matrix.png")
plt.close(fig)
print("fig 2 done")

# =====================================================================
# Association with target (pairwise-complete / NaN-aware Spearman)
# =====================================================================
rho = Xv.corrwith(pd.Series(y, name="y"), method="spearman")
rho_df = pd.DataFrame({
    "rho": rho,
    "abs_rho": rho.abs(),
    "n_pairwise": Xv.notna().sum(),
}).sort_values("abs_rho", ascending=False)
rho_df.rename_axis("column").to_csv(ANALYSIS / "eda_association_spearman.csv")
top25 = rho_df.head(25)
print("\nTop-25 associations:\n", top25["rho"].round(4).to_string())

# =====================================================================
# Figure 3 — top-25 association
# =====================================================================
fig, ax = plt.subplots(figsize=(9.8, 9.4))
names = top25.index.tolist()[::-1]
vals = top25["rho"].tolist()[::-1]
cols = [C_POS if v > 0 else C_NEG for v in vals]
ax.barh(range(len(vals)), vals, color=cols, edgecolor="white", height=0.72)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(names, fontsize=9.5)
for i, v in enumerate(vals):
    ax.text(v + (0.010 if v > 0 else -0.010), i, f"{v:+.3f}",
            va="center", ha="left" if v > 0 else "right", fontsize=8.5,
            color="#7a4a1e" if v > 0 else "#33507a")
ax.axvline(0, color="#333333", lw=1)
ax.set_xlim(-0.55, 0.58)
ax.set_xlabel("Spearman ρ with target (y = 1 Fair/Poor)")
ax.set_title("Top-25 whitelist features by |Spearman ρ| with the target")
ax.text(0.985, 0.972,
        f"n = {n:,} (pairwise complete; sentinels → NaN)\n"
        "no blacklisted column present by construction (ADR-005 R1 hard assert)\n"
        f"orange = positive ρ (higher value → Fair/Poor), blue = negative ρ",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="#F7F7F7", ec="#CCCCCC"))
fig.savefig(FIGURES / "eda_association_top25.png")
plt.close(fig)
print("fig 3 done")

# =====================================================================
# Figure 4 — correlation heatmap of top-40 by |rho|
# =====================================================================
BLOCKS = [
    ("Health days", ["PHYSHLTH", "POORHLTH", "MENTHLTH"]),
    ("Disability", ["DIFFWALK", "DIFFALON", "DIFFDRES", "DECIDE"]),
    ("Socio-demo", ["INCOME3", "EMPLOY1", "EDUCA", "_AGE80"]),
]
block_cols = [c for _, cols in BLOCKS for c in cols]
top40 = rho_df.head(40).index.tolist()
rest = [c for c in top40 if c not in block_cols]
rest = sorted(rest, key=lambda c: -rho[c])  # signed rho desc: positive -> negative
order = block_cols + rest
C40 = Xv[order].corr(method="spearman")

fig, ax = plt.subplots(figsize=(12.2, 10.0))
sns.heatmap(C40, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            square=True, cbar_kws={"label": "Spearman ρ", "shrink": 0.75, "pad": 0.015},
            xticklabels=order, yticklabels=order, linewidths=0.2, linecolor="white")
ax.tick_params(labelsize=7.6)
plt.setp(ax.get_xticklabels(), rotation=90)
plt.setp(ax.get_yticklabels(), rotation=0)

pos = 0
for name, cols in BLOCKS:
    k = len(cols)
    ax.add_patch(Rectangle((pos - 0.5, pos - 0.5), k, k, fill=False, ec=C_TOP, lw=2.6))
    ax.text(pos + (k - 1) / 2, pos + (k - 1) / 2, name, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white",
            path_effects=[patheffects.withStroke(linewidth=2.4, foreground="#3A0D10")])
    pos += k
for sep in [2.5, 6.5, 10.5]:
    ax.axhline(sep, color="#333333", lw=0.9, ls=":")
    ax.axvline(sep, color="#333333", lw=0.9, ls=":")

ax.set_title("Spearman correlation among the top-40 features (by |ρ| with target)")
fig.text(0.5, 0.005,
         f"n = {n:,}, pairwise complete. Crimson frames mark the three named blocks; "
         "remaining 28 features ordered by signed ρ with the target.\n"
         "Blank cells: ρ undefined — HAVARTH5 is constant on those pairwise-complete subsets (arthritis module skip logic).",
         ha="center", fontsize=8.5, color="#444444")
fig.savefig(FIGURES / "eda_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("fig 4 done")

# block internal correlations for the notes
for name, cols in BLOCKS:
    cm = Xv[cols].corr(method="spearman").to_numpy()
    tri = cm[np.triu_indices(len(cols), 1)]
    print(f"block {name}: pairwise rho range [{tri.min():+.3f}, {tri.max():+.3f}]")

# =====================================================================
# Figure 5 — class-conditional distributions
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
y0, y1 = y == 0, y == 1


def dens_panel(ax, col, bins, xlabel, title, note, cut=None, means=True):
    v = Xv[col]
    if cut is not None:
        v = v.clip(upper=cut)
    ax.hist([v[y0].dropna(), v[y1].dropna()], bins=bins, density=True,
            color=[C_NEG, C_POS], alpha=0.75, label=["y = 0", "y = 1"], edgecolor="white")
    if means:
        for m, c in [(v[y0].mean(), C_NEG), (v[y1].mean(), C_POS)]:
            ax.axvline(m, color=c, ls="--", lw=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density (within class)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.985, 0.965, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F7F7F7", ec="#CCCCCC"))


dens_panel(axes[0, 0], "_AGE80", np.arange(18, 82, 2), "Age (years, top-coded at 80)",
           "_AGE80 — age",
           f"mean {Xv['_AGE80'][y0].mean():.1f} vs {Xv['_AGE80'][y1].mean():.1f}\n"
           f"spike at 80: {np.mean(Xv['_AGE80']==80)*100:.1f} % (top-coding)")
dens_panel(axes[0, 1], "PHYSHLTH", np.arange(-0.5, 31.5, 1), "Days of poor physical health (past 30)",
           "PHYSHLTH — poor physical health days",
           f"mean {Xv['PHYSHLTH'][y0].mean():.1f} vs {Xv['PHYSHLTH'][y1].mean():.1f}\n"
           f"zero spike {np.mean(Xv['PHYSHLTH']==0)*100:.1f} % (88 → 0 recode)")
dens_panel(axes[0, 2], "MENTHLTH", np.arange(-0.5, 31.5, 1), "Days of poor mental health (past 30)",
           "MENTHLTH — poor mental health days",
           f"mean {Xv['MENTHLTH'][y0].mean():.1f} vs {Xv['MENTHLTH'][y1].mean():.1f}\n"
           f"zero spike {np.mean(Xv['MENTHLTH']==0)*100:.1f} % (88 → 0 recode)")

# INCOME3 — grouped bars of within-class shares
ax = axes[1, 0]
inc_codes = np.arange(1, 12)
s0 = Xv["INCOME3"][y0].value_counts(normalize=True).reindex(inc_codes).fillna(0) * 100
s1 = Xv["INCOME3"][y1].value_counts(normalize=True).reindex(inc_codes).fillna(0) * 100
w = 0.4
ax.bar(inc_codes - w / 2, s0, width=w, color=C_NEG, label="y = 0", edgecolor="white")
ax.bar(inc_codes + w / 2, s1, width=w, color=C_POS, label="y = 1", edgecolor="white")
ax.plot([Xv["INCOME3"][y0].median()], [ax.get_ylim()[1] * 0.96], marker="v", color=C_NEG, ms=9, clip_on=False)
ax.plot([Xv["INCOME3"][y1].median()], [ax.get_ylim()[1] * 0.96], marker="v", color=C_POS, ms=9, clip_on=False)
ax.set_xticks(inc_codes)
ax.set_xlabel("Income bracket code (1 = lowest <$10k … 11 = highest ≥$200k)")
ax.set_ylabel("Share within class (%)")
ax.set_title("INCOME3 — household income bracket")
ax.legend(frameon=False, fontsize=9)
ax.text(0.985, 0.965, f"medians: {Xv['INCOME3'][y0].median():.0f} (y=0) vs {Xv['INCOME3'][y1].median():.0f} (y=1)\n"
        f"non-missing n = {Xv['INCOME3'].notna().sum():,}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#F7F7F7", ec="#CCCCCC"))

# DIFFWALK — rate panel
ax = axes[1, 1]
r0 = Xv["DIFFWALK"][y0].eq(1).mean() * 100
r1 = Xv["DIFFWALK"][y1].eq(1).mean() * 100
bars = ax.bar([0, 1], [r0, r1], color=[C_NEG, C_POS], width=0.5, edgecolor="white")
for b, r in zip(bars, [r0, r1]):
    ax.text(b.get_x() + b.get_width() / 2, r + 1.2, f"{r:.1f} %", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["y = 0", "y = 1"])
ax.set_ylim(0, 60)
ax.set_ylabel("P(DIFFWALK = 1 | class)  (%)")
ax.set_title("DIFFWALK — serious difficulty walking")
ax.text(0.03, 0.90, f"rate ratio {r1/r0:.1f}×\nnon-missing n = {Xv['DIFFWALK'].notna().sum():,}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#F7F7F7", ec="#CCCCCC"))

# _SMOKER3 — grouped bars of within-class shares
ax = axes[1, 2]
smk_codes = np.arange(1, 5)
smk_lbl = ["Current\nevery day", "Current\nsome days", "Former", "Never"]
s0 = Xv["_SMOKER3"][y0].value_counts(normalize=True).reindex(smk_codes).fillna(0) * 100
s1 = Xv["_SMOKER3"][y1].value_counts(normalize=True).reindex(smk_codes).fillna(0) * 100
ax.bar(smk_codes - w / 2, s0, width=w, color=C_NEG, label="y = 0", edgecolor="white")
ax.bar(smk_codes + w / 2, s1, width=w, color=C_POS, label="y = 1", edgecolor="white")
ax.set_xticks(smk_codes)
ax.set_xticklabels(smk_lbl, fontsize=9)
ax.set_xlabel("Smoker status (computed ordinal _SMOKER3)")
ax.set_ylabel("Share within class (%)")
ax.set_title("_SMOKER3 — smoker status")
ax.legend(frameon=False, fontsize=9)
ax.text(0.985, 0.965, f"current (1+2): {s0[1]+s0[2]:.1f} % (y=0) vs {s1[1]+s1[2]:.1f} % (y=1)\n"
        f"non-missing n = {Xv['_SMOKER3'].notna().sum():,}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#F7F7F7", ec="#CCCCCC"))

fig.suptitle("Class-conditional distributions of key features — y = 1 (Fair/Poor) vs y = 0 (Good or better)",
             fontsize=14, fontweight="bold")
fig.savefig(FIGURES / "eda_key_features_by_class.png")
plt.close(fig)
print("fig 5 done")

# =====================================================================
# extra facts for the notes
# =====================================================================
print("\nEMPLOY1==8 (unable to work) share: y0 {:.3f}% y1 {:.3f}%".format(
    Xv["EMPLOY1"][y0].eq(8).mean() * 100, Xv["EMPLOY1"][y1].eq(8).mean() * 100))
print("EXERANY2==1 (any exercise) share: y0 {:.1f}% y1 {:.1f}%".format(
    Xv["EXERANY2"][y0].eq(1).mean() * 100, Xv["EXERANY2"][y1].eq(1).mean() * 100))
for c in ["FRUIT2", "FRUITJU2", "FVGREEN1", "FRENCHF1", "POTATOE1", "VEGETAB2"]:
    v = Xv[c]
    print(f"{c}: ==0.5 {(v == 0.5).sum():,} | ==0 {(v == 0).sum():,} | max {v.max():.1f} | missing {v.isna().mean()*100:.1f}%")
print("POORHLTH missing {:.1f}% | JOINPAI2 missing {:.1f}%".format(
    Xv["POORHLTH"].isna().mean() * 100, Xv["JOINPAI2"].isna().mean() * 100))
print("CHECKUP1 rho {:.4f} | ALCDAY5 rho {:.4f} | _BMI5 rho {:.4f} | DIABETE4 rho {:.4f}".format(
    rho["CHECKUP1"], rho["ALCDAY5"], rho["_BMI5"], rho["DIABETE4"]))

# interpretive block grouping (feeds P5 hypothesis)
groups = {
    "functional limitation / joint pain": ["DIFFWALK", "DIFFALON", "DIFFDRES", "DECIDE", "JOINPAI2", "LMTJOIN3", "ARTHDIS2"],
    "health days": ["PHYSHLTH", "MENTHLTH", "POORHLTH"],
    "socio-demographics": ["INCOME3", "EDUCA", "_AGE80", "EMPLOY1", "_RACE", "SEXVAR", "MARITAL"],
    "chronic conditions": ["CHCCOPD3", "HAVARTH5", "BPHIGH6", "DIABETE4", "ADDEPEV3", "CHCKDNY2",
                            "CHCSCNCR", "CHCOCNCR", "ASTHMA3", "CVDINFR4", "CVDCRHD4", "CVDSTRK3"],
    "behaviors": ["EXERANY2", "_SMOKER3", "ALCDAY5", "AVEDRNK3", "DRNK3GE5", "MAXDRNKS", "USENOW3", "ECIGNOW1", "_BMI5"],
    "access / utilization": ["PRIMINSR", "PERSDOC3", "MEDCOST1", "CHECKUP1", "FLUSHOT7", "PNEUVAC4", "HIVTST7"],
    "diet (fruit & veg)": ["FRUIT2", "FRUITJU2", "FVGREEN1", "FRENCHF1", "POTATOE1", "VEGETAB2"],
    "admin / geo": ["_STATE", "QSTLANG", "_URBSTAT"],
}
print("\nInterpretive block summary (mean |rho|, mean coverage %):")
for g, cols in groups.items():
    have = [c for c in cols if c in rho.index]
    cov = np.mean([100 - Xv[c].isna().mean() * 100 for c in have])
    print(f"  {g:34s} mean|rho| {rho[have].abs().mean():.3f}  cov {cov:5.1f}  (k={len(have)})")
print("\nALL DONE")
