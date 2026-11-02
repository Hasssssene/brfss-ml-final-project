"""P6 - automated hyperparameter optimization, nested CV, Wilcoxon (EXP-014..015).

Built on the P3/P4 harness. Methodology is ADR-locked and unchanged: y =
GENHLTH in {4,5}, feature set ``all-76`` (437,532 x 76 after the valid-target
mask; 72,741 positives = 16.63%), metric ROC-AUC, RANDOM_STATE=42 is the only
seed anywhere. Every COMPARISON score (tuned finals, nested-CV outer folds,
Wilcoxon pairs) is computed on the canonical folds from
``src.config.make_folds()`` (StratifiedKFold(10, shuffle=True,
random_state=42), fold-index sha256 asserted per row and cross-checked
against every previously logged row).

Four parts:

1. ``search``  RandomizedSearchCV over bounded spaces for the top-2 P4
   families (LightGBM EXP-006 0.903355, XGBoost EXP-007 0.903357):

   - XGBoost: n_estimators ~ randint[300, 800]; learning_rate ~
     loguniform[0.02, 0.15]; max_depth ~ randint[3, 10]; min_child_weight ~
     randint[1, 20]; subsample ~ uniform[0.6, 1.0]; colsample_bytree ~
     uniform[0.5, 1.0]; reg_lambda ~ loguniform[0.1, 10].
   - LightGBM: n_estimators ~ randint[300, 800]; learning_rate ~
     loguniform[0.02, 0.15]; num_leaves ~ randint[15, 127];
     min_child_samples ~ randint[5, 100]; subsample ~ uniform[0.6, 1.0]
     (subsample_freq=1 fixed); colsample_bytree ~ uniform[0.5, 1.0];
     reg_lambda ~ loguniform[0.1, 10].
   - Booster flags fixed as in P4 (XGB tree_method='hist',
     eval_metric='logloss', verbosity=0; LGBM verbosity=-1);
     random_state=42; scipy.stats distributions for continuous params.
   - Search config: RandomizedSearchCV(n_iter=SEARCH_N_ITER,
     cv=StratifiedKFold(5, shuffle=True, random_state=42) - an INNER cv for
     model selection only, NOT the canonical comparison set -,
     scoring='roc_auc', random_state=42, n_jobs=-1, pre_dispatch='n_jobs').
     Fit on FULL X,y: the search itself is legitimate on all data; the
     selection bias this induces is exactly what Part 3 quantifies.
   - Full search tables -> results/p6/search_cv_{xgb,lgbm}.csv (params,
     mean_test_auc, std_test_auc, rank); winners ->
     results/p6/best_params.json.

2. ``finals`` the two winners re-scored on the CANONICAL folds:
   EXP-014a = tuned LGBM, EXP-014b = tuned XGB (NaN-passthrough preprocessor
   inside the pipeline, identical folds, imbalance_strategy=none). Together
   with EXP-006/007 these rows are the tuning-improvement evidence.

3. ``nested`` unbiased nested CV on the better family by Part 2: outer loop
   = the canonical make_folds() (10 folds, sha-asserted); inner =
   RandomizedSearchCV(n_iter=NESTED_N_ITER, cv=StratifiedKFold(3,
   shuffle=True, random_state=42), roc_auc, rs=42, refit on the outer
   training fold) per outer fold. EXP-015 logs mean +/- std of the 10 outer
   ROC-AUCs. Optimism of the non-nested tuned estimate = EXP-014x mean -
   EXP-015 mean. Per-fold winners -> results/p6/nested_cv_details.csv.

4. ``wilcoxon`` Wilcoxon signed-rank tests (scipy.stats.wilcoxon, two-sided,
   zero_method='wilcox') on the 10 PAIRED canonical-fold AUCs pulled from
   the append-only log (identical folds by construction):
   T1 tuned LGBM (EXP-014a) vs tuned XGB (EXP-014b);
   T2 tuned winner vs its untuned self (EXP-006 or EXP-007);
   T3 tuned winner vs LogReg EXP-002a (effect-size contrast).
   -> results/p6/wilcoxon_tests.csv.

``figure`` writes figures/p6_tuning_results.png (panel a: per-family search
configs mean+/-std inner AUC with winners starred; panel b: untuned vs tuned
vs nested-CV bars with fold-std error bars). ``summary`` writes
results/p6_summary.md. ``probe`` measures one isolated untuned fit per
family on the exact inner-5-fold row count and prints wall-time projections
(compute budget: target total <= 90 min; sanctioned levers are n_iter
20 -> 15 and/or n_estimators cap 800 -> 600, documented in
BUDGET_ADJUSTMENTS).

Resume-safe like P3-P5: search results and nested per-fold details are
checkpointed to results/p6/ and reused; exp ids already in
results/experiments_log.csv are never re-run.

Usage (one foreground command per part):
    python -m src.p6_tune probe
    python -m src.p6_tune search
    python -m src.p6_tune finals
    python -m src.p6_tune nested
    python -m src.p6_tune wilcoxon
    python -m src.p6_tune figure
    python -m src.p6_tune summary
    python -m src.p6_tune all        # search..summary (probe excluded)

This module never touches git, vault/ or deliverables/, and writes only to
results/ (experiments_log.csv + p6/ + p6_summary.md) and figures/.
"""
from __future__ import annotations

import os

# Thread hygiene: 1 BLAS/OpenMP thread per parallel fit worker (search folds
# and canonical folds run in loky workers; the models are n_jobs=1).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from src.config import FIGURES, RANDOM_STATE, RESULTS  # noqa: E402
from src.experiments import (  # noqa: E402
    _hash_prefix_from_notes,
    load_logged_runs,
    load_xy,
    log_scored_run,
    make_fold_list,
    run_experiment,
)
from src.p4_models import build_nan_passthrough_preprocessor  # noqa: E402

P6_DIR = RESULTS / "p6"
P6_SUMMARY_MD = RESULTS / "p6_summary.md"
BEST_PARAMS_JSON = P6_DIR / "best_params.json"
NESTED_DETAILS_CSV = P6_DIR / "nested_cv_details.csv"
WILCOXON_CSV = P6_DIR / "wilcoxon_tests.csv"
TIMING_JSON = P6_DIR / "timing.json"
FIG_TUNING = FIGURES / "p6_tuning_results.png"

# ---- search configuration (compute-budget levers, see module docstring) ----
# BUDGET DECISION (documented, both levers sanctioned by the P6 task spec):
# the timing probe measured an isolated untuned XGB fit at 60.4s on the
# inner-5-fold row count; combined with P4's logged parallel efficiency for
# XGB (~0.40 on this 12-logical-core box: 138.3s wall for 10 fold fits), the
# XGB-wins-Part-2 worst case projects to ~110 min > the 90-min budget at
# n_iter=20 / n_estimators<=800. n_iter 20->15 (Part 1) + n_estimators cap
# 800->600 (Parts 1+3, same space) bring the pessimistic projection to ~85
# min. Untuned configs use 500 trees, so [300,600] still brackets them.
SEARCH_N_ITER = 15            # Part 1 candidates per family (spec: 20; reduced per budget)
NESTED_N_ITER = 12            # Part 3 inner candidates per outer fold (spec: 12)
SEARCH_INNER_SPLITS = 5       # Part 1 inner CV (model selection only)
NESTED_INNER_SPLITS = 3       # Part 3 inner CV (budget-reduced, per spec)
N_EST_RANGE = (300, 600)      # inclusive n_estimators bounds (spec: [300, 800]; capped per budget)
BUDGET_ADJUSTMENTS: list[str] = [
    "SEARCH_N_ITER 20 -> 15 (Part 1): probe + P4-calibrated projections put "
    "the n_iter=20 worst case over the 90-min budget",
    "n_estimators range [300,800] -> [300,600] (Parts 1+3, same space "
    "shared): same reason; untuned configs (500 trees) stay inside the range",
]

PROJ_CORE_EFFICIENCY = 0.6    # 12 logical cores; measured P4-style scaling
PROJ_HEAVINESS = 1.2          # sampled configs average heavier than untuned


# ---- model constructors (fixed booster flags exactly as P4) -----------------------
def make_lgbm(params: dict) -> LGBMClassifier:
    kw = dict(random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
              subsample_freq=1)
    kw.update(params)
    return LGBMClassifier(**kw)


def make_xgb(params: dict) -> XGBClassifier:
    kw = dict(random_state=RANDOM_STATE, n_jobs=1, tree_method="hist",
              eval_metric="logloss", verbosity=0)
    kw.update(params)
    return XGBClassifier(**kw)


FAMILIES_P6 = {
    "lgbm": {
        "model_name": "LGBMClassifier",
        "make": make_lgbm,
        "untuned_id": "EXP-006",
        "untuned_params": dict(n_estimators=500, learning_rate=0.05,
                               num_leaves=63),
        "tuned_id": "EXP-014a",
    },
    "xgb": {
        "model_name": "XGBClassifier",
        "make": make_xgb,
        "untuned_id": "EXP-007",
        "untuned_params": dict(n_estimators=500, learning_rate=0.05,
                               max_depth=6),
        "tuned_id": "EXP-014b",
    },
}


def search_space(family: str) -> dict:
    """Bounded scipy.stats search space (spec-fixed); fresh dist objects."""
    common = {
        "model__n_estimators": stats.randint(N_EST_RANGE[0], N_EST_RANGE[1] + 1),
        "model__learning_rate": stats.loguniform(0.02, 0.15),
        "model__subsample": stats.uniform(0.6, 0.4),        # [0.6, 1.0]
        "model__colsample_bytree": stats.uniform(0.5, 0.5),  # [0.5, 1.0]
        "model__reg_lambda": stats.loguniform(0.1, 10.0),
    }
    if family == "xgb":
        common.update({
            "model__max_depth": stats.randint(3, 11),        # [3, 10]
            "model__min_child_weight": stats.randint(1, 21),  # [1, 20]
        })
    elif family == "lgbm":
        common.update({
            "model__num_leaves": stats.randint(15, 128),      # [15, 127]
            "model__min_child_samples": stats.randint(5, 101),  # [5, 100]
        })
    else:
        raise ValueError(f"unknown family: {family}")
    return common


def space_desc(family: str) -> str:
    rng = f"[{N_EST_RANGE[0]},{N_EST_RANGE[1]}]"
    d = (f"n_estimators~randint{rng}; learning_rate~logU[0.02,0.15]; "
         f"subsample~U[0.6,1.0]; colsample_bytree~U[0.5,1.0]; "
         f"reg_lambda~logU[0.1,10]")
    if family == "xgb":
        return d + "; max_depth~randint[3,10]; min_child_weight~randint[1,20]"
    return d + ("; num_leaves~randint[15,127]; "
                "min_child_samples~randint[5,100]; subsample_freq=1 fixed")


# ---- small helpers -----------------------------------------------------------------
def _sanitize(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _sanitize_params(d: dict) -> dict:
    return {_sanitize(k): _sanitize(v) for k, v in d.items()}


def _record_timing(cmd: str, wall_s: float) -> None:
    P6_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(TIMING_JSON.read_text(encoding="utf-8")) \
        if TIMING_JSON.exists() else []
    data.append({"cmd": cmd,
                 "timestamp": datetime.now().isoformat(timespec="seconds"),
                 "wall_s": round(float(wall_s), 1)})
    TIMING_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_best_params(required: tuple[str, ...] = ()) -> dict:
    if not BEST_PARAMS_JSON.exists():
        return {}
    best = json.loads(BEST_PARAMS_JSON.read_text(encoding="utf-8"))
    missing = [f for f in required if f not in best]
    if missing:
        raise SystemExit(f"[p6] best_params.json missing families {missing} - "
                         "run `python -m src.p6_tune search` first")
    return best


def _assert_log_folds(logged: dict[str, dict], fold_hash: str) -> None:
    """Every logged row must have been scored on exactly these folds."""
    for exp_id in sorted(logged):
        prior = _hash_prefix_from_notes(logged[exp_id]["notes"])
        if prior and not fold_hash.startswith(prior):
            raise AssertionError(
                f"{exp_id}: logged folds_sha256={prior} != canonical "
                f"{fold_hash[:12]}")
    print(f"[p6] fold identity verified against all {len(logged)} "
          "pre-logged runs (folds_sha256 "
          f"{fold_hash[:12]})")


def _caught_warning_msgs(caught: list) -> list[str]:
    return sorted({f"{w.category.__name__}: {str(w.message)[:300]}"
                   for w in caught})


# ---- Part 0: timing probe & projections --------------------------------------------
def cmd_probe() -> None:
    """One isolated single-thread untuned fit per family + wall projections.

    Fits on the FIRST inner-5-fold training split (exactly the row count the
    Part-1 search fits see) so the measurement matches the workload. This is
    a throwaway timing probe - no model is kept, nothing is logged.
    """
    X, y = load_xy()
    splitter = StratifiedKFold(n_splits=SEARCH_INNER_SPLITS, shuffle=True,
                               random_state=RANDOM_STATE)
    tr, _ = next(iter(splitter.split(X, y)))
    Xtr = X.iloc[tr]
    print(f"[p6] probe: isolated single-thread fit per family on the first "
          f"inner-{SEARCH_INNER_SPLITS}-fold train split ({len(tr):,} rows)")
    fits: dict[str, float] = {}
    for family in ("lgbm", "xgb"):
        spec = FAMILIES_P6[family]
        pipe = Pipeline([
            ("prep", build_nan_passthrough_preprocessor()),
            ("model", spec["make"](spec["untuned_params"])),
        ])
        t0 = time.perf_counter()
        pipe.fit(Xtr, y[tr])
        fits[family] = time.perf_counter() - t0
        print(f"[p6]   {family:5s} untuned-config fit: {fits[family]:.1f}s")

    ncores = os.cpu_count() or 1
    eff_cores = ncores * PROJ_CORE_EFFICIENCY
    inner_ratio = (2 / 3) * 0.9 / 0.8    # nested inner train vs probe rows
    outer_ratio = 0.9 / 0.8              # outer train vs probe rows

    def part1(f):
        return SEARCH_N_ITER * SEARCH_INNER_SPLITS * fits[f] * PROJ_HEAVINESS \
            / eff_cores

    def part3(f):
        per_fold = (NESTED_N_ITER * NESTED_INNER_SPLITS * fits[f]
                    * PROJ_HEAVINESS * inner_ratio
                    + fits[f] * PROJ_HEAVINESS * outer_ratio)
        return 10 * per_fold / eff_cores

    print(f"[p6] projections (ncores={ncores}, assumed efficiency "
          f"{PROJ_CORE_EFFICIENCY:.2f}, heaviness x{PROJ_HEAVINESS:.0f}):")
    for f in ("lgbm", "xgb"):
        print(f"[p6]   {f:5s}: part1 search ~{part1(f) / 60:5.1f} min | "
              f"part3 nested ~{part3(f) / 60:5.1f} min")
    worst_nested = max(part3(f) for f in ("lgbm", "xgb"))
    total = (part1("lgbm") + part1("xgb") + 300.0 + worst_nested + 600.0) / 60
    print(f"[p6] projected worst-case total (part1 both + finals + nested "
          f"worst family + overheads): ~{total:.1f} min "
          f"(budget 90 min; levers: n_iter 20->15, n_estimators cap 600)")


# ---- Part 1: RandomizedSearchCV -----------------------------------------------------
def run_search(family: str, X, y, n_iter: int) -> dict:
    """RandomizedSearchCV on FULL X,y; saves table + best params; returns info.

    Inner cv = StratifiedKFold(5, shuffle=True, random_state=42) for model
    SELECTION ONLY (not the canonical comparison folds). pre_dispatch
    'n_jobs' bounds the number of in-flight X pickles (as in P3/P4).
    """
    spec = FAMILIES_P6[family]
    pipe = Pipeline([
        ("prep", build_nan_passthrough_preprocessor()),
        ("model", spec["make"]({})),
    ])
    search = RandomizedSearchCV(
        pipe, search_space(family), n_iter=n_iter, scoring="roc_auc",
        cv=StratifiedKFold(n_splits=SEARCH_INNER_SPLITS, shuffle=True,
                           random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=-1, pre_dispatch="n_jobs",
        refit=True, verbose=2, error_score="raise", return_train_score=False,
    )
    print(f"[p6] RandomizedSearchCV {family}: n_iter={n_iter}, inner "
          f"StratifiedKFold({SEARCH_INNER_SPLITS}, shuffle, rs=42), "
          f"scoring=roc_auc, n_jobs=-1; space: {space_desc(family)}")
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(X, y)
    wall = time.perf_counter() - t0
    warn_msgs = _caught_warning_msgs(caught)

    cvr = search.cv_results_
    table = pd.DataFrame({
        "params": [json.dumps(_sanitize_params(p), sort_keys=True)
                   for p in cvr["params"]],
        "mean_test_auc": cvr["mean_test_score"].astype(float),
        "std_test_auc": cvr["std_test_score"].astype(float),
        "rank": cvr["rank_test_score"].astype(int),
    }).sort_values("rank", kind="stable").reset_index(drop=True)
    P6_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = P6_DIR / f"search_cv_{family}.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[p6] wrote {csv_path} ({len(table)} candidates); search wall "
          f"{wall / 60:.1f} min")

    best_params = {k.replace("model__", ""): _sanitize(v)
                   for k, v in search.best_params_.items()}
    info = {
        "best_params": best_params,
        "best_inner_mean_auc": float(search.best_score_),
        "best_inner_std_auc": float(cvr["std_test_score"][search.best_index_]),
        "best_rank": int(cvr["rank_test_score"][search.best_index_]),
        "n_candidates": int(n_iter),
        "inner_cv": (f"StratifiedKFold({SEARCH_INNER_SPLITS}, shuffle=True, "
                     f"random_state=42) - model selection only, NOT the "
                     "canonical comparison folds"),
        "fit_on": "full X,y (437,532 rows)",
        "wall_s": round(wall, 1),
        "parent_warnings": warn_msgs,
    }
    best = _load_best_params()
    best[family] = info
    best["search_config"] = {
        "n_iter": n_iter,
        "scoring": "roc_auc",
        "random_state": RANDOM_STATE,
        "inner_cv_splits": SEARCH_INNER_SPLITS,
        "n_estimators_range_inclusive": list(N_EST_RANGE),
        "space": {f: space_desc(f) for f in ("lgbm", "xgb")},
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    if BUDGET_ADJUSTMENTS:
        best["search_config"]["budget_adjustments"] = list(BUDGET_ADJUSTMENTS)
    BEST_PARAMS_JSON.write_text(json.dumps(best, indent=2), encoding="utf-8")

    print(f"[p6] {family} best inner AUC = {info['best_inner_mean_auc']:.6f} "
          f"+/- {info['best_inner_std_auc']:.6f} | params: {best_params}")
    print(f"[p6] {family} top-5 candidates:")
    for _, r in table.head(5).iterrows():
        print(f"      #{int(r['rank']):>2} mean={r['mean_test_auc']:.6f} "
              f"std={r['std_test_auc']:.6f}  {r['params']}")
    if warn_msgs:
        print(f"[p6] {family} parent-side warnings during search: "
              f"{warn_msgs}")
    else:
        print(f"[p6] {family} no warnings captured in the parent process "
              "(worker-side warnings are not transportable - as in P3)")
    return info


def cmd_search() -> None:
    X, y = load_xy()
    best = _load_best_params()
    for family in ("lgbm", "xgb"):
        csv_path = P6_DIR / f"search_cv_{family}.csv"
        if csv_path.exists() and family in best:
            print(f"[p6] search {family}: {csv_path.name} already present - "
                  "skipping (resume)")
            continue
        best[family] = run_search(family, X, y, SEARCH_N_ITER)


# ---- Part 2: tuned finals on the canonical folds ------------------------------------
def cmd_finals() -> None:
    logged = load_logged_runs()
    best = _load_best_params(required=("lgbm", "xgb"))
    X, y = load_xy()
    folds, fold_hash = make_fold_list(X, y)
    _assert_log_folds(logged, fold_hash)

    for family in ("lgbm", "xgb"):
        spec = FAMILIES_P6[family]
        exp_id = spec["tuned_id"]
        if exp_id in logged:
            print(f"[p6] {exp_id} already logged - skipping (resume)")
            continue
        base_id = spec["untuned_id"]
        if base_id not in logged:
            raise SystemExit(f"[p6] {exp_id} needs untuned baseline "
                             f"{base_id} logged first")
        params = best[family]["best_params"]
        params_json = json.dumps(params, sort_keys=True)
        note = (
            f"P6 tuned final: winner of RandomizedSearchCV(n_iter="
            f"{best[family]['n_candidates']}; inner StratifiedKFold("
            f"{SEARCH_INNER_SPLITS}, shuffle, rs=42) - selection only; "
            f"roc_auc; rs=42; n_jobs=-1) fit on FULL X,y (selection bias of "
            f"that non-nested protocol is quantified by nested CV EXP-015); "
            f"space: {space_desc(family)}; untuned baseline {base_id} = "
            f"{logged[base_id]['mean']:.6f}; tables results/p6/"
            f"search_cv_{family}.csv + results/p6/best_params.json"
        )
        r = run_experiment(
            exp_id, spec["make"](params), spec["model_name"], params_json,
            X, y, note=note, folds=folds, fold_hash=fold_hash,
            preprocessor=build_nan_passthrough_preprocessor(),
        )
        delta = r["mean"] - logged[base_id]["mean"]
        print(f"[p6] {exp_id} tuned {spec['model_name']} vs untuned "
              f"{base_id}: {r['mean']:.6f} vs "
              f"{logged[base_id]['mean']:.6f} (delta {delta:+.6f})")


# ---- Part 3: nested CV (unbiased estimate of the tuned pipeline) --------------------
def cmd_nested() -> None:
    logged = load_logged_runs()
    for e in ("EXP-014a", "EXP-014b"):
        if e not in logged:
            raise SystemExit(f"[p6] nested needs {e} logged first "
                             "(run `python -m src.p6_tune finals`)")
    if "EXP-015" in logged:
        print("[p6] EXP-015 already logged - skipping (resume)")
        return
    # better family by Part 2 (deterministic tie-break: LGBM - cheaper)
    better = "lgbm" if (logged["EXP-014a"]["mean"]
                        >= logged["EXP-014b"]["mean"]) else "xgb"
    spec = FAMILIES_P6[better]
    tuned_id = spec["tuned_id"]
    print(f"[p6] nested family = {better} ({spec['model_name']}); Part-2 "
          f"means: EXP-014a {logged['EXP-014a']['mean']:.6f} vs EXP-014b "
          f"{logged['EXP-014b']['mean']:.6f}")

    X, y = load_xy()
    folds, fold_hash = make_fold_list(X, y)   # outer loop = canonical folds
    _assert_log_folds(logged, fold_hash)
    space = search_space(better)

    rows: dict[int, dict] = {}
    if NESTED_DETAILS_CSV.exists():
        prior = pd.read_csv(NESTED_DETAILS_CSV)
        if {"family", "fold"}.issubset(prior.columns) and \
                (prior["family"] == better).any():
            prior = prior[prior["family"] == better]
            for _, rr in prior.iterrows():
                rows[int(rr["fold"]) - 1] = rr.to_dict()
            print(f"[p6] nested resume: reusing {len(rows)} outer fold(s) "
                  f"from {NESTED_DETAILS_CSV.name} (family={better})")

    for i, (tr, te) in enumerate(folds):
        if i in rows:
            print(f"[p6] outer fold {i + 1}/10 already computed - reusing "
                  f"auc={rows[i]['outer_test_auc']:.6f}")
            continue
        search = RandomizedSearchCV(
            Pipeline([("prep", build_nan_passthrough_preprocessor()),
                      ("model", spec["make"]({}))]),
            space, n_iter=NESTED_N_ITER, scoring="roc_auc",
            cv=StratifiedKFold(n_splits=NESTED_INNER_SPLITS, shuffle=True,
                               random_state=RANDOM_STATE),
            random_state=RANDOM_STATE, n_jobs=-1, pre_dispatch="n_jobs",
            refit=True, verbose=1, error_score="raise",
            return_train_score=False,
        )
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X.iloc[tr], y[tr])
        fold_wall = time.perf_counter() - t0
        warn_msgs = _caught_warning_msgs(caught)
        proba = search.predict_proba(X.iloc[te])[:, 1]
        auc = float(roc_auc_score(y[te], proba))
        cvr = search.cv_results_
        rows[i] = {
            "family": better,
            "fold": i + 1,
            "n_train_rows": int(len(tr)),
            "best_params": json.dumps(
                _sanitize_params(search.best_params_), sort_keys=True),
            "best_inner_mean_auc": float(search.best_score_),
            "best_inner_std_auc": float(
                cvr["std_test_score"][search.best_index_]),
            "outer_test_auc": auc,
            "wall_s": round(fold_wall, 1),
            "parent_warnings": " | ".join(warn_msgs),
        }
        out = pd.DataFrame([rows[k] for k in sorted(rows)])
        out.to_csv(NESTED_DETAILS_CSV, index=False, encoding="utf-8")
        print(f"[p6] outer fold {i + 1}/10: outer AUC={auc:.6f} | inner best "
              f"{rows[i]['best_inner_mean_auc']:.6f} | wall "
              f"{fold_wall / 60:.1f} min | params {rows[i]['best_params']}")

    outer = np.array([float(rows[i]["outer_test_auc"]) for i in range(10)])
    wall_total = float(sum(float(rows[i]["wall_s"]) for i in range(10)))
    optimism = logged[tuned_id]["mean"] - float(outer.mean())
    params_desc = json.dumps({
        "family": better,
        "estimator": spec["model_name"],
        "search": (f"RandomizedSearchCV(n_iter={NESTED_N_ITER}; cv="
                   f"StratifiedKFold({NESTED_INNER_SPLITS}, shuffle=True, "
                   f"random_state=42); scoring=roc_auc; random_state=42; "
                   "refit per outer fold)"),
        "space": space_desc(better),
        "per_fold_best_params": "results/p6/nested_cv_details.csv",
    }, sort_keys=True)
    log_scored_run(
        "EXP-015", outer, spec["model_name"], params_desc,
        note=(f"P6 nested CV (Advanced): outer loop = the canonical 10 folds "
              f"(folds_sha256 {fold_hash[:12]} asserted - same folds as every "
              f"other run); inner = RandomizedSearchCV(n_iter="
              f"{NESTED_N_ITER}; 3-fold inner StratifiedKFold(shuffle, rs=42)"
              f"; roc_auc; rs=42) per outer fold, best config refit on the "
              f"outer training fold; UNBIASED estimate of the tuned "
              f"{spec['model_name']} pipeline; family = Part-2 winner "
              f"(EXP-014a {logged['EXP-014a']['mean']:.6f} vs EXP-014b "
              f"{logged['EXP-014b']['mean']:.6f}); optimism of non-nested "
              f"{tuned_id} estimate = {optimism:+.6f}; details "
              f"results/p6/nested_cv_details.csv"),
        runtime_s=wall_total, fold_hash=fold_hash,
    )
    print(f"[p6] nested unbiased estimate: {outer.mean():.6f} +/- "
          f"{outer.std(ddof=0):.6f} | non-nested tuned {tuned_id} = "
          f"{logged[tuned_id]['mean']:.6f} | optimism {optimism:+.6f} | "
          f"untuned {spec['untuned_id']} = "
          f"{logged[spec['untuned_id']]['mean']:.6f}")


# ---- Part 4: Wilcoxon signed-rank tests ---------------------------------------------
def _wilcoxon_row(tid: str, description: str, a_id: str, b_id: str,
                  runs: dict[str, dict]) -> dict:
    """One two-sided Wilcoxon signed-rank test on paired canonical-fold AUCs."""
    a, b = runs[a_id]["scores"], runs[b_id]["scores"]
    d = a - b
    res = stats.wilcoxon(a, b)          # two-sided, zero_method='wilcox'
    # scipy's WilcoxonResult does not expose which p-value method it used;
    # derive it from its documented rule: exact when n<=25 and no zeros
    # (dropped by 'wilcox') and no ties among |nonzero diffs|, else normal
    # approximation.
    nz = d[d != 0.0]
    has_ties = len(np.unique(np.abs(nz))) != len(nz)
    method = ("exact" if (len(nz) <= 25 and not has_ties)
              else "asymptotic (normal approximation)")
    med = float(np.median(d))
    wins_a, wins_b, ties = int((d > 0).sum()), int((d < 0).sum()), \
        int((d == 0).sum())
    direction = (f"{a_id} > {b_id}" if med > 0 else
                 f"{a_id} < {b_id}" if med < 0 else "no median difference")
    sig = bool(res.pvalue < 0.05)
    if sig:
        verdict = (f"Significant at alpha=0.05: {direction} "
                   f"(median paired delta {med:+.6f} AUC per fold, "
                   f"{wins_a}/{len(d)} folds favour {a_id}).")
    else:
        verdict = (f"Not significant at alpha=0.05 "
                   f"(p={res.pvalue:.4f}): the paired fold difference "
                   f"(median {med:+.6f}) is within what fold noise "
                   "produces; treat the two as statistically "
                   "indistinguishable on these 10 folds.")
    return {
        "test_id": tid,
        "comparison": description,
        "a_exp_id": a_id,
        "b_exp_id": b_id,
        "model_a": runs[a_id]["model"],
        "model_b": runs[b_id]["model"],
        "mean_auc_a": float(a.mean()),
        "mean_auc_b": float(b.mean()),
        "delta_mean_auc": float(d.mean()),
        "n_pairs": int(len(d)),
        "folds_a_greater": wins_a,
        "folds_b_greater": wins_b,
        "ties": ties,
        "median_paired_delta": med,
        "mean_paired_delta": float(d.mean()),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "method": method,
        "alternative": "two-sided",
        "zero_method": "wilcox",
        "direction": direction,
        "significant_0_05": sig,
        "verdict": verdict,
    }


def cmd_wilcoxon() -> None:
    logged = load_logged_runs()
    needed = ["EXP-014a", "EXP-014b", "EXP-006", "EXP-007", "EXP-002a"]
    missing = [e for e in needed if e not in logged]
    if missing:
        raise SystemExit(f"[p6] wilcoxon needs {missing} logged first")
    X, y = load_xy()   # only to rebuild + assert the canonical fold identity
    _, fold_hash = make_fold_list(X, y)
    _assert_log_folds(logged, fold_hash)

    better = "lgbm" if (logged["EXP-014a"]["mean"]
                        >= logged["EXP-014b"]["mean"]) else "xgb"
    spec = FAMILIES_P6[better]
    records = [
        _wilcoxon_row(
            "T1", "tuned LightGBM vs tuned XGBoost (both P6 winners)",
            "EXP-014a", "EXP-014b", logged),
        _wilcoxon_row(
            "T2", f"tuned {spec['model_name']} vs its untuned self "
                  f"({spec['untuned_id']})", spec["tuned_id"],
            spec["untuned_id"], logged),
        _wilcoxon_row(
            "T3", "tuned best booster vs LogisticRegression EXP-002a "
                  "(effect-size contrast)", spec["tuned_id"], "EXP-002a",
            logged),
    ]
    P6_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(WILCOXON_CSV, index=False, encoding="utf-8")
    print(f"[p6] wrote {WILCOXON_CSV}")
    for r in records:
        print(f"[p6] {r['test_id']}: W={r['statistic']:.1f} "
              f"p={r['p_value']:.6g} ({r['method']}) | direction "
              f"{r['direction']} | median paired delta "
              f"{r['median_paired_delta']:+.6f} | "
              f"{'SIGNIFICANT' if r['significant_0_05'] else 'not significant'}"
              f" at 0.05")


# ---- figure -------------------------------------------------------------------------
def cmd_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logged = load_logged_runs()
    for e in ("EXP-014a", "EXP-014b", "EXP-006", "EXP-007"):
        if e not in logged:
            raise SystemExit(f"[p6] figure needs {e} logged first")
    fold_prefix = (_hash_prefix_from_notes(logged["EXP-006"]["notes"])
                   or "unknown")

    colors = {"lgbm": "#2ca02c", "xgb": "#1f77b4"}
    labels = {"lgbm": "LightGBM", "xgb": "XGBoost"}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # ---- panel a: search candidates (mean +/- std inner 5-fold AUC) ----
    for j, family in enumerate(("lgbm", "xgb")):
        tab = pd.read_csv(P6_DIR / f"search_cv_{family}.csv")
        tab = tab.sort_values("rank").head(20).reset_index(drop=True)
        x = np.arange(1, len(tab) + 1) + (0.18 if family == "lgbm" else -0.18)
        ax1.errorbar(x, tab["mean_test_auc"], yerr=tab["std_test_auc"],
                     fmt="o", ms=5, lw=1.1, capsize=2.5, alpha=0.9,
                     color=colors[family], label=labels[family])
        # winner = rank 1
        w = tab.index[tab["rank"] == 1][0]
        ax1.plot(x[w], tab.loc[w, "mean_test_auc"], marker="*", ms=17,
                 mfc="#ffbf00", mec="black", mew=0.7, zorder=5,
                 label=f"{labels[family]} winner")
        for k in range(len(tab)):
            if int(tab.loc[k, "rank"]) <= 5:
                ax1.annotate(str(int(tab.loc[k, "rank"])),
                             (x[k], tab.loc[k, "mean_test_auc"]),
                             xytext=(0, 6), textcoords="offset points",
                             ha="center", fontsize=7, color="#555555")
    ax1.set_xlabel("search candidate (ordered by mean inner 5-fold CV AUC; "
                   "numbers = rank)")
    ax1.set_ylabel("mean inner CV ROC-AUC (+/- std over 5 folds)")
    ax1.set_title("(a) RandomizedSearchCV candidates per family\n"
                  f"n_iter={SEARCH_N_ITER}, inner "
                  f"StratifiedKFold({SEARCH_INNER_SPLITS}, rs=42) - "
                  "selection only", fontsize=10)
    ax1.grid(linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8, loc="lower left")
    ymin = min(pd.read_csv(P6_DIR / f"search_cv_{f}.csv")["mean_test_auc"].min()
               - 3 * pd.read_csv(P6_DIR / f"search_cv_{f}.csv")["std_test_auc"].max()
               for f in ("lgbm", "xgb"))
    ymax = max(pd.read_csv(P6_DIR / f"search_cv_{f}.csv")["mean_test_auc"].max()
               + 3 * pd.read_csv(P6_DIR / f"search_cv_{f}.csv")["std_test_auc"].max()
               for f in ("lgbm", "xgb"))
    ax1.set_ylim(ymin - 0.002, ymax + 0.002)

    # ---- panel b: untuned vs tuned vs nested-CV estimates ----
    bars = [
        ("LGBM untuned\n(EXP-006)", logged["EXP-006"], colors["lgbm"], None),
        ("LGBM tuned\n(EXP-014a)", logged["EXP-014a"], colors["lgbm"], None),
        ("XGB untuned\n(EXP-007)", logged["EXP-007"], colors["xgb"], None),
        ("XGB tuned\n(EXP-014b)", logged["EXP-014b"], colors["xgb"], None),
    ]
    if "EXP-015" in logged:
        better = "lgbm" if (logged["EXP-014a"]["mean"]
                            >= logged["EXP-014b"]["mean"]) else "xgb"
        bars.append((f"{labels[better]} tuned, nested CV\n(EXP-015, unbiased)",
                     logged["EXP-015"], colors[better], "///"))
    xs = np.arange(len(bars))
    means = [b[1]["mean"] for b in bars]
    stds = [b[1]["std"] for b in bars]
    ax2.bar(xs, means, yerr=stds, width=0.62, capsize=4,
            color=[b[2] for b in bars], edgecolor="black", linewidth=0.6,
            hatch=[b[3] or "" for b in bars], alpha=0.9)
    for x_, m_, s_ in zip(xs, means, stds):
        ax2.text(x_, m_ + s_ + 0.0006, f"{m_:.4f}", ha="center", fontsize=8.5)
    ax2.set_xticks(xs, [b[0] for b in bars], fontsize=8)
    lo = min(m - s for m, s in zip(means, stds))
    ax2.set_ylim(lo - 0.004, max(means) + 0.004)
    ax2.set_ylabel("mean CV ROC-AUC over the 10 canonical folds "
                   "(error bars = fold std)")
    ax2.set_title("(b) tuning effect + nested-CV honesty check\n"
                  "identical canonical folds (sha "
                  f"{fold_prefix}); no resampling", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("P6 hyperparameter optimization - RandomizedSearchCV, tuned "
                 "finals and nested-CV estimate (BRFSS 2021 GENHLTH, all-76)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_TUNING, dpi=200)
    plt.close(fig)
    print(f"[p6] wrote {FIG_TUNING}")


# ---- summary ------------------------------------------------------------------------
def cmd_summary() -> None:
    logged = load_logged_runs()
    needed = ["EXP-014a", "EXP-014b", "EXP-015", "EXP-006", "EXP-007",
              "EXP-002a"]
    missing = [e for e in needed if e not in logged]
    if missing:
        raise SystemExit(f"[p6] summary needs {missing} logged first")
    best = _load_best_params(required=("lgbm", "xgb"))
    wx = pd.read_csv(WILCOXON_CSV)
    nested = pd.read_csv(NESTED_DETAILS_CSV)
    timing = (json.loads(TIMING_JSON.read_text(encoding="utf-8"))
              if TIMING_JSON.exists() else [])

    better = "lgbm" if (logged["EXP-014a"]["mean"]
                        >= logged["EXP-014b"]["mean"]) else "xgb"
    spec = FAMILIES_P6[better]
    tuned_id = spec["tuned_id"]
    optimism = logged[tuned_id]["mean"] - logged["EXP-015"]["mean"]

    def m(e):
        return logged[e]["mean"]

    def s(e):
        return logged[e]["std"]

    lines = [
        "# P6 — Hyperparameter optimization, nested CV, Wilcoxon "
        "(EXP-014a/b, EXP-015)",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} by "
        "`src/p6_tune.py` on the P3/P4 harness. Methodology unchanged "
        "(ADR-locked): y = GENHLTH ∈ {4,5}; feature set `all-76`; ROC-AUC; "
        "no resampling (`imbalance_strategy=none`); NaN-passthrough "
        "preprocessor inside every pipeline (fit on training data only); "
        "RANDOM_STATE=42 everywhere (searches, inner CVs, models).",
        "",
        "## Setup",
        "",
        f"- **Comparison folds**: the canonical "
        "`StratifiedKFold(10, shuffle=True, random_state=42)` via "
        "`src.config.make_folds()` — fold-index sha256 "
        f"`{_hash_prefix_from_notes(logged['EXP-006']['notes'])}` asserted in "
        "every P6 comparison run (EXP-014a/b scored via `run_experiment`, "
        "EXP-015 outer loop = the same fold list) and cross-checked against "
        "all previously logged rows.",
        f"- **Inner CVs (selection only, never comparison)**: Part 1 "
        f"`StratifiedKFold({SEARCH_INNER_SPLITS}, shuffle=True, "
        f"random_state=42)`; Part 3 `StratifiedKFold("
        f"{NESTED_INNER_SPLITS}, shuffle=True, random_state=42)`.",
        f"- **Search**: `RandomizedSearchCV(n_iter={SEARCH_N_ITER}, "
        "scoring='roc_auc', random_state=42, n_jobs=-1, pre_dispatch="
        "'n_jobs')` fit on FULL X,y — legitimate for selection; the "
        "selection bias it induces is exactly what EXP-015 quantifies.",
        "- Spaces (scipy.stats distributions): LightGBM — " +
        space_desc("lgbm") + ". XGBoost — " + space_desc("xgb") + ".",
        f"- Fixed booster flags as in P4: LGBM `verbosity=-1`, "
        "`subsample_freq=1`, `n_jobs=1`, `random_state=42`; XGB "
        "`tree_method='hist'`, `eval_metric='logloss'`, `verbosity=0`, "
        "`n_jobs=1`, `random_state=42`.",
        "",
        "## Part 1 — search winners (top-5 per family, inner 5-fold AUC)",
        "",
    ]
    for family in ("lgbm", "xgb"):
        tab = pd.read_csv(P6_DIR / f"search_cv_{family}.csv") \
            .sort_values("rank")
        lines.append(f"**{FAMILIES_P6[family]['model_name']}** "
                     f"(best inner AUC {best[family]['best_inner_mean_auc']:.6f}"
                     f" ± {best[family]['best_inner_std_auc']:.6f}; search "
                     f"wall {best[family]['wall_s'] / 60:.1f} min):")
        lines.append("")
        lines.append("| rank | mean inner AUC ± std | params |")
        lines.append("|---|---|---|")
        for _, r in tab.head(5).iterrows():
            lines.append(f"| {int(r['rank'])} | {r['mean_test_auc']:.6f} ± "
                         f"{r['std_test_auc']:.6f} | `{r['params']}` |")
        lines.append("")
    lines += [
        "Best params per family (verbatim from `results/p6/best_params.json`):",
        "",
        f"- LightGBM: `{json.dumps(best['lgbm']['best_params'], sort_keys=True)}`",
        f"- XGBoost: `{json.dumps(best['xgb']['best_params'], sort_keys=True)}`",
        "",
        "## Part 2 — tuned finals vs untuned baselines (canonical folds)",
        "",
        "| family | untuned (P4) | mean AUC | tuned (P6) | mean AUC | Δ AUC | "
        "paired folds better |",
        "|---|---|---|---|---|---|---|",
    ]
    for family in ("lgbm", "xgb"):
        sp = FAMILIES_P6[family]
        d = logged[sp["tuned_id"]]["scores"] - logged[sp["untuned_id"]]["scores"]
        lines.append(
            f"| {sp['model_name']} | {sp['untuned_id']} | "
            f"{m(sp['untuned_id']):.6f} ± {s(sp['untuned_id']):.6f} | "
            f"{sp['tuned_id']} | **{m(sp['tuned_id']):.6f} ± "
            f"{s(sp['tuned_id']):.6f}** | {m(sp['tuned_id']) - m(sp['untuned_id']):+.6f} "
            f"| {int((d > 0).sum())}/10 |")
    lines += [
        "",
        "## Part 3 — nested CV: the honest estimate (EXP-015)",
        "",
        f"- Family nested: **{spec['model_name']}** (Part-2 winner: EXP-014a "
        f"{m('EXP-014a'):.6f} vs EXP-014b {m('EXP-014b'):.6f}).",
        f"- Protocol: outer = the 10 canonical folds; inner = "
        f"`RandomizedSearchCV(n_iter={NESTED_N_ITER}, "
        f"StratifiedKFold({NESTED_INNER_SPLITS}, shuffle=True, "
        "random_state=42), roc_auc)` per outer fold, best config refit on "
        "the outer training fold, scored on the untouched outer test fold.",
        f"- **Unbiased tuned estimate: {m('EXP-015'):.6f} ± "
        f"{s('EXP-015'):.6f}** (outer fold AUCs "
        + " ".join(f"{v:.4f}" for v in logged["EXP-015"]["scores"]) + ").",
        f"- **Optimism of the non-nested tuned estimate: "
        f"{optimism:+.6f} AUC** ({tuned_id} {m(tuned_id):.6f} − EXP-015 "
        f"{m('EXP-015'):.6f}) — the Part-2 number overstates generalization "
        "by this much because the same data informed both the search and "
        "the score.",
        f"- Context: nested estimate vs untuned {spec['untuned_id']} "
        f"({m(spec['untuned_id']):.6f}): "
        f"{m('EXP-015') - m(spec['untuned_id']):+.6f}.",
        f"- Per-outer-fold inner winners differ across folds — full table in "
        "`results/p6/nested_cv_details.csv` (search selection is "
        "fold-dependent; the nested protocol prices that in).",
        "",
        "## Part 4 — Wilcoxon signed-rank tests (paired canonical folds)",
        "",
        "| test | comparison | mean AUC a vs b | median paired Δ | W | "
        "p-value | method | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in wx.iterrows():
        lines.append(
            f"| {r['test_id']} | {r['comparison']} | "
            f"{r['mean_auc_a']:.6f} vs {r['mean_auc_b']:.6f} | "
            f"{r['median_paired_delta']:+.6f} | {r['statistic']:.1f} | "
            f"{r['p_value']:.6g} | {r['method']} | {r['verdict']} |")
    lines += [
        "",
        "Plain-language reading:",
        "",
        f"- **T1** ({wx.loc[wx['test_id'] == 'T1', 'verdict'].iloc[0]})",
        f"- **T2** ({wx.loc[wx['test_id'] == 'T2', 'verdict'].iloc[0]})",
        f"- **T3** ({wx.loc[wx['test_id'] == 'T3', 'verdict'].iloc[0]})",
        "",
        "All three tests are two-sided Wilcoxon signed-rank tests "
        "(`scipy.stats.wilcoxon`, `zero_method='wilcox'`) on the 10 paired "
        "canonical-fold AUCs; the folds are identical by construction "
        "(hash-asserted), so pairing is exact.",
        "",
        "## Budget, warnings & artifacts",
        "",
        f"- Compute budget target ≤ 90 min. " +
        ("No reductions applied: n_iter="
         f"{SEARCH_N_ITER} (Part 1) and n_estimators range "
         f"[{N_EST_RANGE[0]},{N_EST_RANGE[1]}] as specified."
         if not BUDGET_ADJUSTMENTS else
         "Adjustments applied (documented): " + "; ".join(BUDGET_ADJUSTMENTS)),
        f"- Parent-process warnings during searches: LightGBM — "
        f"{best['lgbm']['parent_warnings'] or 'none'}; XGBoost — "
        f"{best['xgb']['parent_warnings'] or 'none'}. (Warnings raised "
        "inside loky workers cannot be transported to the parent — same "
        "caveat as P3; boosters emit none at these verbosity settings.)",
        f"- Command walls (results/p6/timing.json): "
        + ("; ".join(f"{t['cmd']} {t['wall_s'] / 60:.1f} min"
                     for t in timing) if timing else "n/a") + ".",
        "- Artifacts: `results/p6/search_cv_lgbm.csv`, "
        "`search_cv_xgb.csv`, `best_params.json`, `nested_cv_details.csv`, "
        "`wilcoxon_tests.csv`, `timing.json`; figure "
        "`figures/p6_tuning_results.png` (dpi 200).",
        f"- Environment: sklearn RandomizedSearchCV; scipy "
        f"{__import__('scipy').__version__}; xgboost "
        f"{__import__('xgboost').__version__}; lightgbm "
        f"{__import__('lightgbm').__version__}; {os.cpu_count()} logical "
        "cores.",
        "",
    ]
    P6_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[p6] wrote {P6_SUMMARY_MD}")
    print(f"[p6] tuned winner: {spec['model_name']} {tuned_id} "
          f"{m(tuned_id):.6f}; nested EXP-015 {m('EXP-015'):.6f}; "
          f"optimism {optimism:+.6f}")


# ---- CLI ---------------------------------------------------------------------------
COMMANDS = ("probe", "search", "finals", "nested", "wilcoxon", "figure",
            "summary")
_FUNCS = {"probe": cmd_probe, "search": cmd_search, "finals": cmd_finals,
          "nested": cmd_nested, "wilcoxon": cmd_wilcoxon, "figure": cmd_figure,
          "summary": cmd_summary}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="P6 hyperparameter optimization + nested CV + Wilcoxon "
                    "(EXP-014a/b, EXP-015)")
    ap.add_argument("targets", nargs="+",
                    choices=list(COMMANDS) + ["all"])
    args = ap.parse_args(argv)
    order = (["search", "finals", "nested", "wilcoxon", "figure", "summary"]
             if "all" in args.targets
             else [c for c in COMMANDS if c in args.targets])
    for cmd in order:
        t0 = time.perf_counter()
        print(f"\n[p6] ===== {cmd} =====")
        _FUNCS[cmd]()
        _record_timing(cmd, time.perf_counter() - t0)
    print(f"\n[p6] done; session wall logged to {TIMING_JSON}")


if __name__ == "__main__":
    main()
