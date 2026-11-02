"""P7b - Case-profile figure + threshold sweep for the interpretive error-analysis note.

Outputs
  figures/p7_case_profiles.png        (dpi 150; 3 panels, Cases A/B/C)
  stdout: every number quoted in results/analysis/error_analysis.md

Read-only inputs (no fits, no randomness, no experiments_log row):
  results/p7/selected_cases.json       - case vectors + population mean/std/z-scores
  results/p7/shap_global_importance.csv- global mean|SHAP| ranking (defines top-12)
  results/p7/shap_cases.parquet        - per-case SHAP (log-odds, 76 original columns)
  results/p7/error_candidates.csv      - top-50 most-confident shared errors (FN subtype)
  data/interim/oof_predictions.parquet - OOF probs (P7 artifact, provenance in
                                         results/p7/oof_provenance.json)

Author: data-analyst, 2026-11-02 (P7b).
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------
# 1. Load artifacts
# ----------------------------------------------------------------------------
cases_json = json.loads((ROOT / "results/p7/selected_cases.json").read_text())
shap_glob = pd.read_csv(ROOT / "results/p7/shap_global_importance.csv")
shap_cases = pd.read_parquet(ROOT / "results/p7/shap_cases.parquet")

TOP = 12
top_feats = shap_glob.sort_values("rank").head(TOP)["column"].tolist()

GENHLTH_LABEL = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

# Codebook-friendly renderings of raw values (BRFSS 2021 codes).
INCOME3_MAP = {
    1: "<$10k", 2: "$10-15k", 3: "$15-20k", 4: "$20-25k", 5: "$25-35k",
    6: "$35-50k", 7: "$50-75k", 8: "$75-100k", 9: "$100-150k", 10: "$150-200k",
    11: ">=$200k",
}
EDUCA_MAP = {
    1: "never attended", 2: "elementary", 3: "some high school",
    4: "high-school grad", 5: "some college", 6: "college grad",
}


def fmt_value(feat: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing (NaN)"
    if feat in ("PHYSHLTH", "MENTHLTH", "POORHLTH"):
        return f"{v:.0f} days"
    if feat == "_BMI5":
        return f"{v:.1f} kg/m2"
    if feat == "DIFFWALK":
        return {1: "1 = yes, difficulty", 2: "2 = no"}[int(v)]
    if feat == "DIABETE4":
        return {1: "1 = yes", 2: "2 = yes, preg. only", 3: "3 = no",
                4: "4 = pre-diabetes"}[int(v)]
    if feat == "EXERANY2":
        return {1: "1 = yes, exercised", 2: "2 = no"}[int(v)]
    if feat == "BPHIGH6":
        return {1: "1 = yes", 2: "2 = no", 3: "3 = borderline"}[int(v)]
    if feat == "SEXVAR":
        return {1: "1 = male", 2: "2 = female"}[int(v)]
    if feat == "_RACE":
        return {1: "1 = White", 2: "2 = Black", 3: "3 = Am. Indian",
                4: "4 = Asian", 5: "5 = Haw./Pac.Isl.", 6: "6 = other",
                7: "7 = multiracial", 8: "8 = Hispanic"}[int(v)]
    if feat == "INCOME3":
        return f"{int(v)}/11 ({INCOME3_MAP[int(v)]})"
    if feat == "EDUCA":
        return f"{int(v)}/6 ({EDUCA_MAP[int(v)]})"
    return f"{v:g}"


# ----------------------------------------------------------------------------
# 2. Figure: population z-profiles of the top-12 SHAP features, per case
# ----------------------------------------------------------------------------
COL_POS = "#C0392B"   # SHAP > 0: pushes toward y = 1 (Fair/Poor)
COL_NEG = "#2471A3"   # SHAP < 0: pushes toward y = 0 (Excellent/Good)
CASE_TITLES = {
    "A": ("Case A - row 307041 (fold 4) | most confident FALSE POSITIVE",
          "y=0, GENHLTH=3 'Good' | OOF P(Fair/Poor): LGBM 0.960 / XGB 0.982 / LR 0.993 | wrong in 3/3 models"),
    "B": ("Case B - row 392245 (fold 9) | most confident FALSE NEGATIVE",
          "y=1, GENHLTH=4 'Fair' | OOF P(Fair/Poor): LGBM 0.0025 / XGB 0.0027 / LR 0.0156 | wrong in 3/3 models"),
    "C": ("Case C - row 208256 (fold 4) | shared error closest to the 3/4 boundary",
          "y=0, GENHLTH=3 'Good' | OOF P(Fair/Poor): LGBM 0.5000 / XGB 0.598 / LR 0.563 | wrong in 3/3 models"),
}

feat_cols = [c for c in shap_cases.columns if c not in ("_base_value", "_row_index")]
ypos = np.arange(TOP)[::-1]  # rank 1 at top

fig, axes = plt.subplots(1, 3, figsize=(17.5, 8.6), sharey=True)

for ax, case_id in zip(axes, ["A", "B", "C"]):
    case = cases_json["cases"][case_id]
    sv = shap_cases.loc[case_id]

    for i, feat in enumerate(top_feats):
        val = case["features"][feat]["value"]
        z = case["features"][feat]["z"]
        s = float(sv[feat])
        color = COL_POS if s > 0 else COL_NEG
        label = f"{fmt_value(feat, val)}  |  SHAP {s:+.2f}"

        if z is None:
            ax.text(0.15, ypos[i],
                    f"missing (NaN)  |  SHAP {s:+.2f}  (z undefined)",
                    va="center", ha="left", fontsize=8, style="italic", color="0.35")
            continue
        ax.barh(ypos[i], z, height=0.62, color=color, alpha=0.88,
                edgecolor="white", linewidth=0.5, zorder=3)
        # anchor labels at the bar end for positive z, just right of the zero
        # line for negative z (keeps long labels inside the panel)
        x_text = z + 0.15 if z >= 0 else 0.15
        ax.text(x_text, ypos[i], label, va="center", ha="left",
                fontsize=8, color="0.15")

    ax.axvline(0, color="0.4", lw=1, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.set_xlim(-6.5, 8.5)
    ax.set_ylim(-0.7, TOP - 0.3)
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.9", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("population z-score of the case's value  (z = (value - mean) / std, all 437,532 rows)",
                  fontsize=9)

    t1, t2 = CASE_TITLES[case_id]
    ax.set_title(f"{t1}\n{t2}", fontsize=10, loc="left")

    total_shap = float(sv[feat_cols].sum())
    logit = float(sv["_base_value"]) + total_shap
    p_refit = 1.0 / (1.0 + np.exp(-logit))
    ax.text(0.5, -0.145,
            f"SHAP decomposition (final-model refit, log-odds):\n"
            f"base {sv['_base_value']:.3f} + sum(SHAP) {total_shap:+.2f} = {logit:+.2f}"
            f"  ->  refit P(y=1) = {p_refit:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="0.25")

fig.suptitle(
    "P7b - Anatomy of three failure modes: where each case sits in the population on the top-12 "
    "mean|SHAP| features (tuned LightGBM, EXP-014a)",
    fontsize=13, fontweight="bold", x=0.02, ha="left")

legend = fig.legend(
    handles=[Patch(facecolor=COL_POS, label="SHAP > 0: this feature pushes the case toward y=1 (Fair/Poor)"),
             Patch(facecolor=COL_NEG, label="SHAP < 0: this feature pushes the case toward y=0 (Excellent/Good)")],
    loc="upper right", ncol=1, fontsize=9, frameon=True,
    bbox_to_anchor=(0.995, 0.985))

fig.text(
    0.02, 0.012,
    "How to read: bar length = how unusual the case's raw value is vs the full valid sample (z, ddof=0, NaN-aware; "
    "results/p7/selected_cases.json).\n"
    "Bar color = the SIGN of that feature's SHAP value for this case in the final-model refit "
    "(results/p7/shap_cases.parquet, TreeExplainer path-dependent, log-odds).\n"
    "Caution: yes/no items are coded 1 = yes / 2 = no, so for them a NEGATIVE z means the condition IS present "
    "(e.g. DIFFWALK = 1 'yes' -> z = -2.28). Labels sit at the bar end (z > 0) or just right of the zero line (z < 0).\n"
    "Feature order = global mean|SHAP| rank 1-12 (results/p7/shap_global_importance.csv): "
    + ", ".join(top_feats) + ".\n"
    "OOF probabilities come from the 10-fold CV models (data/interim/oof_predictions.parquet); "
    "SHAP values from the full-data refit proxy - two different objects by construction.",
    fontsize=8, color="0.3", va="bottom")

fig.subplots_adjust(left=0.085, right=0.985, top=0.845, bottom=0.235, wspace=0.16)
out = ROOT / "figures/p7_case_profiles.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"[saved] {out.relative_to(ROOT)}")

# ----------------------------------------------------------------------------
# 3. Verified numbers for the note (stdout)
# ----------------------------------------------------------------------------
print("\n=== top-12 features (global mean|SHAP| order) ===")
print(shap_glob.sort_values("rank").head(TOP)[["rank", "column", "mean_abs_shap"]].to_string(index=False))

print("\n=== per-case SHAP sums (refit proxy, log-odds) ===")
for case_id in ["A", "B", "C"]:
    sv = shap_cases.loc[case_id]
    tot = float(sv[feat_cols].sum())
    logit = float(sv["_base_value"]) + tot
    print(f"case {case_id}: sum(SHAP) {tot:+.4f}  logit {logit:+.4f}  refit p {1/(1+np.exp(-logit)):.4f}")

print("\n=== case-level SHAP on selected features ===")
sel = ["PHYSHLTH", "POORHLTH", "DIFFWALK", "_BMI5", "DIABETE4", "INCOME3",
       "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKDNY2", "EDUCA", "MENTHLTH", "EMPLOY1"]
print(shap_cases[sel].round(4).to_string())

# --- error_candidates.csv: the confident-FN subtype --------------------------
ec = pd.read_csv(ROOT / "results/p7/error_candidates.csv")
print(f"\n=== error_candidates.csv: n={len(ec)} rows ===")
print("genhlth_raw mix:", ec["genhlth_raw"].value_counts().to_dict())
print(f"PHYSHLTH == 0: {(ec.PHYSHLTH == 0).sum()}/{len(ec)} | <= 1 day: {(ec.PHYSHLTH <= 1).sum()}/{len(ec)}")
print(f"DIFFWALK == 2 (no difficulty): {(ec.DIFFWALK == 2).sum()}/{len(ec)}")
print(f"DIABETE4 == 3 (no diabetes): {(ec.DIABETE4 == 3).sum()}/{len(ec)}")
print(f"EXERANY2 == 1 (exercised): {(ec.EXERANY2 == 1).sum()}/{len(ec)}")
print(f"_BMI5 mean {ec._BMI5.mean():.2f} (min {ec._BMI5.min():.2f}, max {ec._BMI5.max():.2f})")
print(f"INCOME3 >= 7 ($50k+): {(ec.INCOME3 >= 7).sum()}/{len(ec)} | median bracket {ec.INCOME3.median():.0f}/11")
print(f"EDUCA in (5,6): {ec.EDUCA.isin([5, 6]).sum()}/{len(ec)}")
print(f"MENTHLTH == 0: {(ec.MENTHLTH == 0).sum()}/{len(ec)} | POORHLTH NaN: {ec.POORHLTH.isna().sum()}/{len(ec)}, ==0: {(ec.POORHLTH == 0).sum()}")

# --- threshold sweep on the OOF artifact -------------------------------------
oof = pd.read_parquet(ROOT / "data/interim/oof_predictions.parquet")
p = oof["oof_lgbm"].to_numpy(dtype=float)
y = oof["y_true"].to_numpy(dtype=int)
P, N = int(y.sum()), int((1 - y).sum())
print(f"\n=== OOF threshold sweep (lgbm; positives {P}, negatives {N}) ===")


def row_at_tau(tau):
    tp = int(((p >= tau) & (y == 1)).sum())
    fp = int(((p >= tau) & (y == 0)).sum())
    return (tau, tp / P, fp / N, tp / (tp + fp), tp + fp)


print(f"{'tau':>7} {'TPR':>7} {'FPR':>7} {'precision':>9} {'flagged':>8}")
r = row_at_tau(0.5)
print(f"{0.5:7.4f} {r[1]:7.4f} {r[2]:7.4f} {r[3]:9.4f} {r[4]:8d}")
for target in (0.60, 0.70, 0.80, 0.90):
    k = int(np.ceil(target * P))
    tau = float(np.sort(p[y == 1])[::-1][k - 1])  # k-th largest positive prob
    r = row_at_tau(tau)
    print(f"{tau:7.4f} {r[1]:7.4f} {r[2]:7.4f} {r[3]:9.4f} {r[4]:8d}   (smallest tau with TPR >= {target:.0%})")
