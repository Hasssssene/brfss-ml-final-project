"""EXP-000 — first profiling pass over raw BRFSS 2021 (data/raw/LLCP2021.csv).

Importable module + ``__main__`` runner. Produces:

- ``results/profiling/column_census.csv``      one row per raw column: dtype,
  null/blank counts, exact-match sentinel counts for the 11 BRFSS missing
  codes, n_unique (NaN + sentinels excluded), top-5 values with counts.
- ``results/profiling/target_distribution.csv`` GENHLTH raw code counts
  (1..5, 7, 9, blank) + binary target via ``src.config.make_target``.
- ``results/profiling/association_screen.csv`` Spearman rho vs binary target y
  for every column EXCEPT the Tier-1 blacklist (_RFHLTH, _PHYS14D, _MENT14D);
  sentinels and blanks masked to NaN before ranking; sorted by |rho|.
- ``results/profiling/leakage_probe.csv``      agreement between _RFHLTH and
  the GENHLTH-derived target (documents that _RFHLTH IS the target).
- ``figures/exp000_target_distribution.png``, ``figures/exp000_missingness_top40.png``,
  ``figures/exp000_association_top20.png`` (PNG, dpi=150).
- Appends the EXP-000 row to ``results/experiments_log.csv`` (creates header
  if the file does not exist yet; skips append if EXP-000 already logged).

Methodological notes
--------------------
* Sentinels are counted by EXACT value match against
  {7, 77, 777, 7777, 77755} ∪ {9, 99, 999, 9999, 99999, 999999} — no range
  assumptions. A column whose *legitimate* code equals one of these numbers is
  still counted (each code has its own census column so triage can judge);
  the missingness figure therefore slightly overstates true missingness for
  such columns.
* ``88`` ("none" for day-count variables) is NOT treated as a sentinel, per
  the project data-integrity rules; the 88->0 recode is an ENG decision for
  the triage note, not for profiling.
* In the association screen, sentinel values and blanks are masked to NaN.
  88 stays as-is (it is a real answer), which dampens rho for day-count
  columns relative to the future ENG recode — noted, not "fixed" here.

No randomness is used anywhere in this module.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make `src.config` importable both as a package module and when this file is
# executed directly (python src/profile_data.py).
_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from src.config import (  # noqa: E402
    DATA_RAW,
    EXPERIMENTS_LOG,
    FIGURES,
    RESULTS,
    TARGET_COL,
    make_target,
)

# ---- constants -------------------------------------------------------------
# BRFSS missing sentinels (Don't know = 7-family, Refused = 9-family).
SENTINEL_CODES: tuple[int, ...] = (
    7, 77, 777, 7777, 77777,
    9, 99, 999, 9999, 99999, 999999,
)
# Tier-1 leakage blacklist (vault/40 Data Notes/Leakage watchlist.md).
TIER1_BLACKLIST: tuple[str, ...] = ("_RFHLTH", "_PHYS14D", "_MENT14D")
# Tier-3 construct-suspicion columns highlighted in the association figure.
TIER3_WATCH: tuple[str, ...] = ("POORHLTH", "PHYSHLTH", "MENTHLTH")
EXP_ID = "EXP-000"
PROFILING_DIR = RESULTS / "profiling"
DPI = 150


# ---- helpers ----------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Read the immutable raw CSV with dtype inference in a single pass."""
    print(f"[exp000] reading {DATA_RAW} ...")
    df = pd.read_csv(DATA_RAW, low_memory=False)
    print(f"[exp000] shape = {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def _fmt_val(v) -> str:
    """Compact string for a value (int-like floats -> '88' not '88.0')."""
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else f"{v:.6g}"
    return str(v)


def _as_numeric(series: pd.Series) -> tuple[pd.Series, str]:
    """Coerce a column to numeric (sentinels untouched, NaN preserved).

    Object-dtype columns are coerced with ``pd.to_numeric``; if fewer than
    half of the non-null entries parse numerically, fall back to a
    deterministic ordinal code (sorted unique strings -> 1..k) so Spearman
    still sees a monotone orderable variable.  Returns (numeric_series, note).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float), str(series.dtype)
    parsed = pd.to_numeric(series, errors="coerce")
    if parsed.notna().sum() >= 0.5 * max(series.notna().sum(), 1):
        return parsed, "object->numeric"
    order = {v: i + 1 for i, v in enumerate(sorted(series.dropna().astype(str).unique()))}
    return series.astype(str).map(order).astype(float), "object->ordinal(sorted)"


def _mask_sentinels(num: pd.Series) -> pd.Series:
    """Exact-match sentinel values -> NaN (blanks are already NaN)."""
    return num.where(~num.isin(list(SENTINEL_CODES)))


# ---- deliverable builders ----------------------------------------------------
def build_column_census(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile: dtype, nulls, per-code sentinel counts, uniques, top-5.

    ``n_unique_excl_nan_sentinels`` and ``top5_values_counts`` both exclude
    NaN and exact-match sentinel values; sentinel prevalence is reported in
    its own per-code columns.
    """
    n_rows = len(df)
    records = []
    for col in df.columns:
        s = df[col]
        n_nonnull = int(s.notna().sum())
        counts = {int(c): int((s == c).sum()) for c in SENTINEL_CODES}
        # values that are neither NaN nor exact-match sentinels
        s_real = s[~s.isin(list(SENTINEL_CODES))].dropna()
        vc = s_real.value_counts()
        top5 = "; ".join(f"{_fmt_val(v)}:{int(c)}" for v, c in vc.head(5).items())
        records.append(
            {
                "col_name": col,
                "dtype": str(s.dtype),
                "n_rows": n_rows,
                "n_nonnull": n_nonnull,
                "n_blank": n_rows - n_nonnull,
                "pct_blank": 100.0 * (n_rows - n_nonnull) / n_rows,
                **{f"n_sent_{c}": counts[int(c)] for c in SENTINEL_CODES},
                "n_sentinel_total": int(sum(counts.values())),
                "pct_blank_plus_sentinel": 100.0
                * ((n_rows - n_nonnull) + sum(counts.values()))
                / n_rows,
                "n_unique_excl_nan_sentinels": int(s_real.nunique()),
                "top5_values_counts": top5,
            }
        )
    return pd.DataFrame(records)


def build_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """GENHLTH raw code counts + binary target counts/rate via make_target."""
    g = df[TARGET_COL]
    n_rows = len(df)
    y, valid = make_target(g)
    n_valid = int(valid.sum())

    labels = {
        1: "Excellent",
        2: "Very good",
        3: "Good",
        4: "Fair",
        5: "Poor",
        7: "Don't know (sentinel)",
        9: "Refused (sentinel)",
        "blank": "Not asked / missing",
    }
    rows = []
    for code in [1, 2, 3, 4, 5, 7, 9, "blank"]:
        n = int(g.isna().sum()) if code == "blank" else int((g == code).sum())
        rows.append(
            {
                "code": code,
                "label": labels[code],
                "n": n,
                "pct": 100.0 * n / n_rows,
                "base": "all_rows",
            }
        )
    n_pos = int(y[valid].sum())
    rows += [
        {"code": "n_valid_target", "label": "rows with valid target (GENHLTH 1-5)",
         "n": n_valid, "pct": 100.0 * n_valid / n_rows, "base": "all_rows"},
        {"code": "y=0", "label": "GENHLTH in {1,2,3} (Excellent/Very good/Good)",
         "n": n_valid - n_pos, "pct": 100.0 * (n_valid - n_pos) / n_valid, "base": "valid_target"},
        {"code": "y=1", "label": "GENHLTH in {4,5} (Fair/Poor)",
         "n": n_pos, "pct": 100.0 * n_pos / n_valid, "base": "valid_target"},
        {"code": "binary_rate_y1", "label": "y=1 rate among valid target rows",
         "n": n_pos, "pct": 100.0 * n_pos / n_valid, "base": "valid_target"},
        {"code": "n_dropped_invalid", "label": "rows dropped (GENHLTH 7/9/blank)",
         "n": n_rows - n_valid, "pct": 100.0 * (n_rows - n_valid) / n_rows, "base": "all_rows"},
    ]
    return pd.DataFrame(rows)


def build_association_screen(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho vs binary y for every non-blacklisted column.

    Sentinels and blanks -> NaN before ranking; y comes from
    ``config.make_target`` restricted to valid-target rows. ``n_used`` is the
    number of rows contributing to each rho. 88 ("none" day counts) is left
    as a real value (see module docstring).
    """
    y, valid = make_target(df[TARGET_COL])
    y_valid = y[valid].astype(float)
    rows = []
    cols = [c for c in df.columns if c not in TIER1_BLACKLIST]
    for i, col in enumerate(cols, 1):
        num, note = _as_numeric(df[col])
        num = _mask_sentinels(num)
        both = pd.concat([num[valid], y_valid], axis=1, join="inner").dropna()
        n_used = len(both)
        rho = both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman") if n_used > 1 else np.nan
        rows.append(
            {
                "col_name": col,
                "dtype_used": note,
                "n_used": n_used,
                "spearman_rho": rho,
                "abs_rho": abs(rho) if pd.notna(rho) else np.nan,
            }
        )
        if i % 50 == 0:
            print(f"[exp000] association: {i}/{len(cols)} columns done")
    out = pd.DataFrame(rows).sort_values("abs_rho", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def build_leakage_probe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """_RFHLTH vs target agreement + GENHLTH x _RFHLTH crosstab.

    BRFSS 2021 codebook: _RFHLTH 1 = "Good or Better Health",
    2 = "Fair or Poor Health".  The codebook recode is therefore
    _RFHLTH==2 -> y=1.  The reversed mapping (1 -> y=1) is also reported so
    the direction is verified empirically, not assumed.
    """
    y, valid = make_target(df[TARGET_COL])
    rf = df["_RFHLTH"]
    both = valid & rf.isin([1, 2])
    n_both = int(both.sum())
    agree_codebook = int(((rf == 2).astype(int)[both] == y[both]).sum())
    agree_reversed = int(((rf == 1).astype(int)[both] == y[both]).sum())
    ct = pd.crosstab(df.loc[valid, TARGET_COL], rf[valid], dropna=False)
    ct.index.name = TARGET_COL
    ct.columns.name = "_RFHLTH"
    metrics = [
        ("n_rows_total", len(df)),
        ("n_genhlth_valid", int(valid.sum())),
        ("n_rfhlth_nonnull", int(rf.notna().sum())),
        ("n_both_present", n_both),
        ("n_agree_codebook_recode (_RFHLTH==2 <-> y==1)", agree_codebook),
        ("agreement_rate_codebook_recode", 100.0 * agree_codebook / n_both if n_both else np.nan),
        ("n_agree_reversed_mapping (_RFHLTH==1 <-> y==1)", agree_reversed),
        ("agreement_rate_reversed_mapping", 100.0 * agree_reversed / n_both if n_both else np.nan),
        ("pct_rfhlth_eq_1_of_both (Good or Better)", 100.0 * int((rf[both] == 1).sum()) / n_both if n_both else np.nan),
        ("pct_rfhlth_eq_2_of_both (Fair or Poor)", 100.0 * int((rf[both] == 2).sum()) / n_both if n_both else np.nan),
        ("pct_y1_from_genhlth_on_both_rows", 100.0 * int(y[both].sum()) / n_both if n_both else np.nan),
        ("note", "codebook: _RFHLTH 1=Good or Better, 2=Fair or Poor; hence 2<->y=1"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"]), ct


# ---- figures -----------------------------------------------------------------
def plot_target_distribution(dist: pd.DataFrame, path: Path) -> None:
    raw = dist[dist["base"] == "all_rows"].iloc[:8]
    labels = [f"{c}\n{l}" for c, l in zip(raw["code"], raw["label"])]
    colors = ["#2e7d32", "#4caf50", "#81c784", "#f57c00", "#d32f2f", "#9e9e9e", "#757575", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(len(raw)), raw["n"], color=colors)
    for b, n in zip(bars, raw["n"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{n:,}",
                ha="center", va="bottom", fontsize=8)
    y1 = dist[dist["code"] == "y=1"].iloc[0]
    rate = dist[dist["code"] == "binary_rate_y1"].iloc[0]["pct"]
    ax.set_xticks(range(len(raw)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("respondents")
    ax.set_title(f"{EXP_ID} — GENHLTH raw distribution (BRFSS 2021, n={int(raw['n'].sum()):,})")
    ax.text(0.985, 0.97,
            f"binary target y=1 (Fair/Poor): n={int(y1['n']):,}\n"
            f"rate among valid targets: {rate:.2f}%  (expected ~16.6%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#fff3e0", edgecolor="#f57c00"))
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_missingness_top40(census: pd.DataFrame, path: Path) -> None:
    top = census.nlargest(40, "pct_blank_plus_sentinel").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(top["col_name"], top["pct_blank_plus_sentinel"], color="#5c6bc0")
    ax.set_xlim(0, 105)
    ax.set_xlabel("% blank + sentinel-coded (exact match on 7/9/77/99/... codes)")
    ax.set_title(f"{EXP_ID} — top 40 columns by missingness\n"
                 "NB: legit codes equal to a sentinel value inflate this (see census)",
                 fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_association_top20(screen: pd.DataFrame, path: Path) -> None:
    top = screen.dropna(subset=["abs_rho"]).head(20).iloc[::-1]
    colors = ["#ef6c00" if c in TIER3_WATCH else "#3949ab" for c in top["col_name"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["col_name"], top["spearman_rho"], color=colors)
    for y_pos, (_, r) in zip(range(len(top)), top.iterrows()):
        ax.text(r["spearman_rho"] + (0.01 if r["spearman_rho"] >= 0 else -0.01),
                y_pos, f"{r['spearman_rho']:+.3f}", va="center",
                ha="left" if r["spearman_rho"] >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman rho vs y (GENHLTH in {4,5}), sentinels -> NaN")
    ax.set_title(f"{EXP_ID} — top 20 |Spearman| association with target\n"
                 f"(Tier-1 blacklist excluded; orange = Tier-3 watchlist {list(TIER3_WATCH)})",
                 fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---- experiments log -----------------------------------------------------------
def append_exp000_to_log() -> None:
    """Create results/experiments_log.csv with header if missing; add EXP-000 once."""
    folds = [f"roc_auc_fold{i:02d}" for i in range(1, 11)]
    header = ["exp_id", "timestamp", "model", "feature_set", "imbalance_strategy",
              "params", *folds, "roc_auc_mean", "notes"]
    row = {
        "exp_id": EXP_ID,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "none (data profiling)",
        "feature_set": "all raw columns",
        "imbalance_strategy": "n/a",
        "params": "n/a",
        "roc_auc_mean": "",
        "notes": "profiling run; see results/profiling/",
    }
    if EXPERIMENTS_LOG.exists():
        existing = pd.read_csv(EXPERIMENTS_LOG, dtype=str)
        if (existing["exp_id"] == EXP_ID).any():
            print(f"[exp000] {EXPERIMENTS_LOG} already contains {EXP_ID}; not appending again")
            return
        out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        out.to_csv(EXPERIMENTS_LOG, index=False)
    else:
        pd.DataFrame([row], columns=header).to_csv(EXPERIMENTS_LOG, index=False)
    print(f"[exp000] appended {EXP_ID} row to {EXPERIMENTS_LOG}")


# ---- main ----------------------------------------------------------------------
def main() -> None:
    t0 = datetime.now()
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[exp000] start {t0.isoformat(timespec='seconds')}")

    df = load_raw()
    print(f"[exp000] memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    # 1. column census
    census = build_column_census(df)
    census.to_csv(PROFILING_DIR / "column_census.csv", index=False)
    print(f"[exp000] wrote column_census.csv ({len(census)} rows)")

    # 2. target distribution
    dist = build_target_distribution(df)
    dist.to_csv(PROFILING_DIR / "target_distribution.csv", index=False)
    rate = dist[dist["code"] == "binary_rate_y1"].iloc[0]["pct"]
    print(f"[exp000] binary y=1 rate among valid targets: {rate:.2f}%")

    # 3. association screen (excludes Tier-1 blacklist)
    screen = build_association_screen(df)
    screen.to_csv(PROFILING_DIR / "association_screen.csv", index=False)
    print(f"[exp000] wrote association_screen.csv ({len(screen)} rows)")

    # 4. leakage probe
    probe, crosstab = build_leakage_probe(df)
    probe.to_csv(PROFILING_DIR / "leakage_probe.csv", index=False)
    print("[exp000] GENHLTH x _RFHLTH crosstab (valid rows):")
    print(crosstab.to_string())

    # 5. figures
    plot_target_distribution(dist, FIGURES / "exp000_target_distribution.png")
    plot_missingness_top40(census, FIGURES / "exp000_missingness_top40.png")
    plot_association_top20(screen, FIGURES / "exp000_association_top20.png")
    print("[exp000] wrote 3 figures to figures/")

    # 6. experiments log
    append_exp000_to_log()

    # quick console summary for the run log
    over90 = census[census["pct_blank_plus_sentinel"] > 90]
    print(f"[exp000] columns with >90% blank+sentinel: {len(over90)}")
    t3 = screen[screen["col_name"].isin(TIER3_WATCH)]
    print("[exp000] Tier-3 watch columns (for leakage ADR):")
    print(t3.to_string(index=False))
    print(f"[exp000] done in {(datetime.now() - t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
