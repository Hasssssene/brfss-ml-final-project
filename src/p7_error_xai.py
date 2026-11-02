"""P7a - error analysis + SHAP XAI on the FINAL MODEL (tuned LightGBM, EXP-014a).

This phase is COMPUTATION on already-logged configurations, not a CV
comparison: NO rows are appended to results/experiments_log.csv. The OOF
prediction run below is a deterministic consequence of three logged runs -

- EXP-014a  tuned LGBMClassifier (results/p6/best_params.json -> lgbm),
            NaN-passthrough preprocessor, random_state=42, n_jobs=1
- EXP-014b  tuned XGBClassifier  (results/p6/best_params.json -> xgb),
            same preprocessor, random_state=42, n_jobs=1
- EXP-002a  LogisticRegression C=0.01 (max_iter=2000, lbfgs,
            StandardScaler) on the shared impute preprocessor

- re-fit per canonical fold (StratifiedKFold(10, shuffle=True,
random_state=42) via src.config.make_folds(), fold-index sha256 asserted
and cross-checked against every logged row). Each model's per-fold OOF AUC
is re-computed and asserted against the logged fold scores, so the OOF
matrix is verifiably the same fits that produced EXP-014a/b / EXP-002a.

Parts (one foreground command each; ``all`` runs them in order):

1. ``oof``     fit the three models per canonical fold, predict_proba on
   the held-out fold -> data/interim/oof_predictions.parquet
   [oof_lgbm, oof_xgb, oof_lr, y_true, fold] + results/p7/oof_provenance.json.
   Fold fits run in a joblib Parallel pool (10 workers, model n_jobs=1 per
   worker - same shape as the cross_val_score(n_jobs=-1) harness).

2. ``errors``  error taxonomy of the FINAL MODEL (tuned LGBM) at threshold
   0.5: TP/FP/TN/FN counts + rates overall and broken down by raw GENHLTH
   class 1..5 (does error mass concentrate at the 3/4 boundary?);
   cross-model wrongness (wrong in how many of {LGBM, XGB, LR});
   results/p7/error_candidates.csv = instances wrong in ALL 3 models,
   sorted by |oof_lgbm - 0.5| desc, top 50, with the P5 top-10 importance
   feature values; results/p7/selected_cases.json = 3 diverse cases
   (A most-confident FP, B most-confident FN, C shared error closest to
   the 3/4 boundary) with full feature vectors + population z-scores.

3. ``shap``    fit the tuned LGBM ONCE on ALL data (final-model proxy -
   documented; the graded CV estimate for this configuration remains
   EXP-014a/EXP-015), shap.TreeExplainer (feature_perturbation default
   'tree_path_dependent' - chosen for report speed; interventional needs a
   background sample and is slower), SHAP values on a stratified random
   sample of 20,000 rows (random_state=42). One-hot SHAP contributions are
   AGGREGATED back to the 76 ORIGINAL columns (sum over each column's
   dummy group) so every plot lives in original feature space.
   -> results/p7/shap_global_importance.csv, results/p7/shap_cases.parquet,
      figures/p7_shap_summary.png (beeswarm top 20),
      figures/p7_shap_vs_gain.png (mean|SHAP| rank vs P5 gain rank,
      Spearman annotated), figures/p7_case_{A,B,C}_waterfall.png.

4. ``summary`` results/p7_summary.md from the artifacts above.

SHAP quirks, recorded honestly: TreeExplainer default is path-dependent
(importance credit can spread across correlated one-hot dummies - exactly
why contributions are summed back to original columns); values are in the
raw-margin (log-odds) space; additivity (base + sum of SHAP == model raw
score) is asserted on the sample and on each case row.

This module never touches git, vault/ or deliverables/, and never appends
to results/experiments_log.csv.
"""
from __future__ import annotations

import os

# Thread hygiene: 1 BLAS/OpenMP thread per parallel fold worker (same as
# src/p6_tune.py; the models themselves are n_jobs=1).
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
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from src.config import DATA_INTERIM, FIGURES, RANDOM_STATE, RESULTS  # noqa: E402
from src.experiments import (  # noqa: E402
    TARGET_PARQUET,
    _hash_prefix_from_notes,
    build_preprocessor,
    load_column_groups,
    load_logged_runs,
    load_xy,
    make_fold_list,
)
from src.p4_models import build_nan_passthrough_preprocessor  # noqa: E402
from src.p6_tune import BEST_PARAMS_JSON, make_lgbm, make_xgb  # noqa: E402

P7_DIR = RESULTS / "p7"
P7_SUMMARY_MD = RESULTS / "p7_summary.md"
OOF_PARQUET = DATA_INTERIM / "oof_predictions.parquet"
PROVENANCE_JSON = P7_DIR / "oof_provenance.json"
TAXONOMY_CSV = P7_DIR / "error_taxonomy.csv"
CANDIDATES_CSV = P7_DIR / "error_candidates.csv"
CASES_JSON = P7_DIR / "selected_cases.json"
SHAP_IMPORTANCE_CSV = P7_DIR / "shap_global_importance.csv"
SHAP_CASES_PARQUET = P7_DIR / "shap_cases.parquet"
TIMING_JSON = P7_DIR / "timing.json"
P5_LGBM_IMPORTANCE_CSV = RESULTS / "p5" / "lgbm_importance.csv"

FIG_SHAP_SUMMARY = FIGURES / "p7_shap_summary.png"
FIG_SHAP_VS_GAIN = FIGURES / "p7_shap_vs_gain.png"

THRESHOLD = 0.5
TOP_CANDIDATES = 50
SHAP_SAMPLE_N = 20_000
WATERFALL_MAX_DISPLAY = 14
BEESWARM_MAX_DISPLAY = 20
N_FOLD_WORKERS = 10
LOGGED_AUC_TOL = 1e-5   # OOF-vs-logged per-fold AUC tolerance (expect ~1e-7)

# exp ids whose logged configurations are re-materialized as OOF predictions
OOF_MODEL_IDS = ("EXP-014a", "EXP-014b", "EXP-002a")


# ---- model builders (exact re-materialization of the logged pipelines) -------------
def build_tuned_lgbm_pipe() -> Pipeline:
    """EXP-014a: tuned LGBM + NaN-passthrough preprocessor."""
    best = json.loads(BEST_PARAMS_JSON.read_text(encoding="utf-8"))
    params = best["lgbm"]["best_params"]
    return Pipeline([("prep", build_nan_passthrough_preprocessor()),
                     ("model", make_lgbm(params))])


def build_tuned_xgb_pipe() -> Pipeline:
    """EXP-014b: tuned XGB + NaN-passthrough preprocessor."""
    best = json.loads(BEST_PARAMS_JSON.read_text(encoding="utf-8"))
    params = best["xgb"]["best_params"]
    return Pipeline([("prep", build_nan_passthrough_preprocessor()),
                     ("model", make_xgb(params))])


def build_lr_pipe() -> Pipeline:
    """EXP-002a: impute preprocessor + StandardScaler + LR(C=0.01, lbfgs)."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.01, max_iter=2000,
                                   random_state=RANDOM_STATE)),
    ])


OOF_MODELS = {  # module-level builders only (must stay picklable for loky)
    "lgbm": dict(exp_id="EXP-014a", build=build_tuned_lgbm_pipe,
                 desc="tuned LGBMClassifier (EXP-014a config)"),
    "xgb": dict(exp_id="EXP-014b", build=build_tuned_xgb_pipe,
                desc="tuned XGBClassifier (EXP-014b config)"),
    "lr": dict(exp_id="EXP-002a", build=build_lr_pipe,
               desc="LogisticRegression C=0.01 pipeline (EXP-002a config)"),
}


def _fit_predict_oof(build_fn, X, y, tr, te):
    """Fit one pipeline on the 9 training folds, predict_proba on the holdout."""
    pipe = build_fn()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # e.g. ConvergenceWarning off-process
        pipe.fit(X.iloc[tr], y[tr])
    proba = pipe.predict_proba(X.iloc[te])[:, 1].astype(np.float64)
    return np.asarray(te, dtype=np.int64), proba


def _record_timing(cmd: str, wall_s: float) -> None:
    P7_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(TIMING_JSON.read_text(encoding="utf-8")) \
        if TIMING_JSON.exists() else []
    data.append({"cmd": cmd, "timestamp": datetime.now().isoformat(
        timespec="seconds"), "wall_s": round(float(wall_s), 1)})
    TIMING_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _assert_log_folds(logged: dict[str, dict], fold_hash: str) -> None:
    """Every logged row must have been scored on exactly these folds."""
    for exp_id in sorted(logged):
        prior = _hash_prefix_from_notes(logged[exp_id]["notes"])
        if prior and not fold_hash.startswith(prior):
            raise AssertionError(
                f"{exp_id}: logged folds_sha256={prior} != canonical "
                f"{fold_hash[:12]}")
    print(f"[p7] fold identity verified against all {len(logged)} logged "
          f"runs (folds_sha256={fold_hash[:12]})")


# ---- Part 1: out-of-fold predictions on the canonical folds ------------------------
def cmd_oof() -> None:
    logged = load_logged_runs()
    X, y = load_xy()
    folds, fold_hash = make_fold_list(X, y)
    _assert_log_folds(logged, fold_hash)

    n = len(X)
    oof = pd.DataFrame({"y_true": y.astype(np.int64)})
    fold_of_row = np.full(n, -1, dtype=np.int64)
    for i, (_, te) in enumerate(folds):
        fold_of_row[te] = i + 1          # 1-based, matches log fold01..fold10
    assert (fold_of_row > 0).all()

    timings, verification = {}, {}
    for key in ("lgbm", "xgb", "lr"):     # XGB last-longest; order as logged
        spec = OOF_MODELS[key]
        exp_id = spec["exp_id"]
        if exp_id not in logged:
            raise SystemExit(f"[p7] {exp_id} not in experiments_log.csv - "
                             "cannot re-materialize its OOF predictions")
        t0 = time.perf_counter()
        results = Parallel(n_jobs=N_FOLD_WORKERS, pre_dispatch="n_jobs")(
            delayed(_fit_predict_oof)(spec["build"], X, y, tr, te)
            for tr, te in folds)
        col = np.full(n, np.nan)
        for te, proba in results:
            col[te] = proba
        if not np.isfinite(col).all():
            raise RuntimeError(f"[p7] oof_{key}: non-finite predictions")
        oof[f"oof_{key}"] = col
        wall = time.perf_counter() - t0
        timings[key] = round(wall, 1)

        # integrity: per-fold OOF AUC must reproduce the logged fold scores
        fold_aucs = np.array([roc_auc_score(y[te], col[te])
                              for _, te in folds])
        dev = float(np.max(np.abs(fold_aucs - logged[exp_id]["scores"])))
        verification[key] = {
            "exp_id": exp_id,
            "logged_mean_auc": float(logged[exp_id]["mean"]),
            "oof_mean_auc": float(np.mean([roc_auc_score(
                y[te], col[te]) for _, te in folds])),
            "max_abs_per_fold_auc_deviation": dev,
            "matches_logged_run": bool(dev <= LOGGED_AUC_TOL),
        }
        ok = "OK" if dev <= LOGGED_AUC_TOL else "MISMATCH"
        print(f"[p7] oof_{key} ({exp_id}): wall {wall / 60:.1f} min | "
              f"max |OOF-vs-logged fold AUC| = {dev:.2e} [{ok}]")
        if dev > LOGGED_AUC_TOL:
            raise SystemExit(
                f"[p7] OOF predictions for {exp_id} do not reproduce the "
                f"logged fold scores (max dev {dev:.2e}) - aborting rather "
                "than writing unfaithful artifacts")

    oof["fold"] = fold_of_row
    oof.to_parquet(OOF_PARQUET, index=False)
    P7_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "statement": (
            "OOF predictions are a deterministic consequence of the logged "
            "configurations EXP-014a, EXP-014b and EXP-002a (params verbatim "
            "from results/p6/best_params.json and the experiments_log rows); "
            "no new experiments_log.csv row was appended for this "
            "computation - it is not a new CV comparison."),
        "folds": ("StratifiedKFold(n_splits=10, shuffle=True, "
                  "random_state=42) via src.config.make_folds(); "
                  "sha asserted against every logged row"),
        "folds_sha256": fold_hash[:12],
        "models": {k: {"exp_id": v["exp_id"], "desc": v["desc"],
                       "params": logged[v["exp_id"]]["params"]}
                   for k, v in OOF_MODELS.items()},
        "random_state": RANDOM_STATE,
        "fold_column": "1-based fold id (1..10) matching log fold01..fold10",
        "verification": verification,
        "wall_s_per_model": timings,
        "n_rows": int(n),
    }
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2),
                               encoding="utf-8")
    print(f"[p7] wrote {OOF_PARQUET}")
    print(f"[p7] wrote {PROVENANCE_JSON}")
    print(f"[p7] total wall {sum(timings.values()) / 60:.1f} min "
          f"(lgbm {timings['lgbm'] / 60:.1f}, xgb {timings['xgb'] / 60:.1f}, "
          f"lr {timings['lr'] / 60:.1f})")


# ---- Part 2: error taxonomy, cross-model wrongness, candidate selection ------------
def _load_oof_with_context() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray,
                                      np.ndarray]:
    """OOF parquet + X + y + genhlth_raw aligned to X's row order."""
    oof = pd.read_parquet(OOF_PARQUET)
    X, y = load_xy()
    tgt = pd.read_parquet(TARGET_PARQUET)
    valid = tgt["valid"].to_numpy(dtype=bool)
    genhlth = tgt.loc[valid, "genhlth_raw"].reset_index(drop=True) \
        .to_numpy(dtype=np.float64)
    assert len(genhlth) == len(X) == len(oof)
    assert (oof["y_true"].to_numpy(dtype=int) == y).all()
    assert oof["fold"].min() >= 1 and oof["fold"].max() <= 10
    return oof, X, y, genhlth


def _pred(frame_or_array):
    a = frame_or_array if isinstance(frame_or_array, np.ndarray) \
        else frame_or_array.to_numpy()
    return (a >= THRESHOLD).astype(int)


def cmd_errors() -> None:
    P7_DIR.mkdir(parents=True, exist_ok=True)
    oof, X, y, genhlth = _load_oof_with_context()
    p = oof["oof_lgbm"].to_numpy()
    pred = _pred(p)
    gh = genhlth.astype(int)          # 1..5 (7/9 dropped by the valid mask)

    # ---- overall taxonomy ----
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    rows = [{
        "scope": "overall", "genhlth_class": "1-5 (all)", "n": n,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "fp_rate": fp / n_neg, "fn_rate": fn / n_pos,
        "tpr": tp / n_pos, "tnr": tn / n_neg,
        "error_rate": (fp + fn) / n, "prevalence": n_pos / n,
    }]

    # ---- per raw GENHLTH class: where do errors live? ----
    for c in (1, 2, 3, 4, 5):
        m = gh == c
        nc = int(m.sum())
        row = {"scope": "by_genhlth_class", "genhlth_class": c, "n": nc,
               "TP": int(((pred == 1) & (y == 1) & m).sum()),
               "FP": int(((pred == 1) & (y == 0) & m).sum()),
               "TN": int(((pred == 0) & (y == 0) & m).sum()),
               "FN": int(((pred == 0) & (y == 1) & m).sum())}
        if c in (1, 2, 3):   # true negatives: errors are false positives
            row["fp_rate"] = row["FP"] / nc
            row["fn_rate"] = np.nan
            row["tpr"] = np.nan
            row["tnr"] = row["TN"] / nc
        else:                # true positives: errors are false negatives
            row["fp_rate"] = np.nan
            row["fn_rate"] = row["FN"] / nc
            row["tpr"] = row["TP"] / nc
            row["tnr"] = np.nan
        row["error_rate"] = (row["FP"] + row["FN"]) / nc
        row["prevalence"] = np.nan
        rows.append(row)

    # ---- cross-model wrongness ----
    wrong = {m: (_pred(oof[f"oof_{m}"]) != y) for m in ("lgbm", "xgb", "lr")}
    k_wrong = sum(wrong.values()).astype(int)   # bool sum -> ndarray
    shared_all = k_wrong == 3
    for k in (0, 1, 2, 3):
        rows.append({"scope": "wrong_in_k_models", "genhlth_class": k,
                     "n": int((k_wrong == k).sum()), "TP": np.nan,
                     "FP": np.nan, "TN": np.nan, "FN": np.nan,
                     "fp_rate": np.nan, "fn_rate": np.nan, "tpr": np.nan,
                     "tnr": np.nan, "error_rate": np.nan,
                     "prevalence": (k_wrong == k).mean()})
    tax = pd.DataFrame(rows)
    tax.to_csv(TAXONOMY_CSV, index=False, encoding="utf-8")
    print(f"[p7] wrote {TAXONOMY_CSV}")
    print(f"[p7] tuned LGBM @ {THRESHOLD}: TP={tp:,} FP={fp:,} TN={tn:,} "
          f"FN={fn:,} | FPR={fp / n_neg:.4f} FNR={fn / n_pos:.4f} "
          f"err={(fp + fn) / n:.4f}")
    for c in (1, 2, 3):
        r = tax[(tax["scope"] == "by_genhlth_class")
                & (tax["genhlth_class"] == c)].iloc[0]
        print(f"[p7]   class {c}: n={r['n']:,}  FP={int(r['FP']):,} "
              f"(FP rate {r['fp_rate']:.4f})")
    for c in (4, 5):
        r = tax[(tax["scope"] == "by_genhlth_class")
                & (tax["genhlth_class"] == c)].iloc[0]
        print(f"[p7]   class {c}: n={r['n']:,}  FN={int(r['FN']):,} "
              f"(FN rate {r['fn_rate']:.4f})")
    for k in (0, 1, 2, 3):
        print(f"[p7]   wrong in {k}/3 models: {int((k_wrong == k).sum()):,} "
              f"({(k_wrong == k).mean() * 100:.2f}%)")

    # ---- candidates: shared errors, most confident first ----
    imp = pd.read_csv(P5_LGBM_IMPORTANCE_CSV).sort_values("rank")
    top10_imp = imp.head(10)["column"].tolist()
    print(f"[p7] P5 top-10 importance features: {top10_imp}")

    shared_idx = np.flatnonzero(shared_all)
    order = shared_idx[np.argsort(-np.abs(p[shared_idx] - THRESHOLD),
                                  kind="stable")]
    keep = order[:TOP_CANDIDATES]
    cand = pd.DataFrame({
        "row_index": keep,
        "y_true": y[keep],
        "genhlth_raw": genhlth[keep],
        "fold": oof["fold"].to_numpy()[keep],
        "oof_lgbm": p[keep],
        "oof_xgb": oof["oof_xgb"].to_numpy()[keep],
        "oof_lr": oof["oof_lr"].to_numpy()[keep],
        "conf_margin_lgbm": np.abs(p[keep] - THRESHOLD),
    })
    for c in top10_imp:                       # P5 top-10 feature values
        cand[c] = X.iloc[keep][c].to_numpy()
    cand.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8")
    print(f"[p7] wrote {CANDIDATES_CSV} ({len(shared_idx):,} shared-error "
          f"instances total; kept top {len(keep)} by |oof_lgbm-0.5|)")

    # ---- 3 selected cases ----
    fp_mask = (y == 0) & (pred == 1)
    fn_mask = (y == 1) & (pred == 0)
    case_a = int(np.flatnonzero(fp_mask)[np.argmax(p[fp_mask])])
    case_b = int(np.flatnonzero(fn_mask)[np.argmin(p[fn_mask])])
    c_mask = shared_all & np.isin(gh, (3, 4))
    if not c_mask.any():
        raise SystemExit("[p7] no shared-error instance at the 3/4 boundary")
    case_c = int(np.flatnonzero(c_mask)[
        np.argmin(np.abs(p[c_mask] - THRESHOLD))])

    pop_mean = np.nanmean(X.to_numpy(dtype=np.float64), axis=0)
    pop_std = np.nanstd(X.to_numpy(dtype=np.float64), axis=0)  # ddof=0
    pop_miss = np.isnan(X.to_numpy(dtype=np.float64)).sum(axis=0)

    def case_payload(idx: int, rule: str) -> dict:
        vals = X.iloc[idx]
        feats = {}
        for j, col in enumerate(X.columns):
            v = vals[col]
            v = None if pd.isna(v) else float(v)
            z = None
            if v is not None and pop_std[j] > 0:
                z = float((v - pop_mean[j]) / pop_std[j])
            feats[col] = {"value": v, "z": z}
        return {
            "rule": rule,
            "row_index": idx,
            "y_true": int(y[idx]),
            "genhlth_raw": float(genhlth[idx]),
            "fold": int(oof["fold"].to_numpy()[idx]),
            "oof_lgbm": float(p[idx]),
            "oof_xgb": float(oof["oof_xgb"].to_numpy()[idx]),
            "oof_lr": float(oof["oof_lr"].to_numpy()[idx]),
            "wrong_in": {m: bool(wrong[m][idx])
                         for m in ("lgbm", "xgb", "lr")},
            "n_models_wrong": int(k_wrong[idx]),
            "features": feats,
        }

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "threshold": THRESHOLD,
        "model": "tuned LGBMClassifier (EXP-014a config), OOF predictions "
                 "from data/interim/oof_predictions.parquet",
        "population_stats": {
            col: {"mean": float(pop_mean[j]), "std": float(pop_std[j]),
                  "n_missing": int(pop_miss[j])}
            for j, col in enumerate(X.columns)},
        "z_score_note": "z = (value - population mean)/population std over "
                        "all 437,532 rows, NaN-aware (ddof=0); NaN feature "
                        "values get z = null.",
        "cases": {
            "A": case_payload(
                case_a, "most confident FALSE POSITIVE: y=0, highest "
                        f"oof_lgbm among the {int(fp_mask.sum()):,} FPs"),
            "B": case_payload(
                case_b, "most confident FALSE NEGATIVE: y=1, lowest "
                        f"oof_lgbm among the {int(fn_mask.sum()):,} FNs"),
            "C": case_payload(
                case_c, "shared-error instance closest to the 3/4 boundary: "
                        "wrong in all 3 models, genhlth_raw in {3,4}, "
                        "|oof_lgbm - 0.5| minimal"),
        },
        "shared_errors_total": int(shared_all.sum()),
        "top10_importance_features": top10_imp,
    }
    CASES_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[p7] wrote {CASES_JSON}")
    for cid, idx in (("A", case_a), ("B", case_b), ("C", case_c)):
        cp = payload["cases"][cid]
        print(f"[p7]   case {cid}: row {idx} | y={cp['y_true']} genhlth="
              f"{cp['genhlth_raw']:.0f} | p_lgbm={cp['oof_lgbm']:.4f} "
              f"p_xgb={cp['oof_xgb']:.4f} p_lr={cp['oof_lr']:.4f} | "
              f"fold={cp['fold']}")


# ---- Part 3: SHAP on the tuned LGBM (final-model proxy) ---------------------------
def _shap_matrix(explainer, Xt) -> np.ndarray:
    """shap_values with defensive handling of list / 3-class shapes."""
    import shap
    sv = explainer.shap_values(Xt, check_additivity=True)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:      # (n, features, 2) legacy layout -> positive class
        sv = sv[..., -1]
    assert sv.ndim == 2 and sv.shape[0] == len(Xt), sv.shape
    return sv.astype(np.float64)


def _base_value(explainer) -> float:
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)):
        return float(np.ravel(ev)[-1])
    return float(ev)


def _original_space_groups(prep, X) -> list[list[int]]:
    """Structural map: transformed column indices per ORIGINAL X column.

    nom block first (OneHotEncoder names '<col>_<cat>'), then the
    passthrough ORD/NUM/ENG block. Asserts the groups partition all
    transformed columns exactly once and there are exactly 76 groups.
    """
    groups = load_column_groups()
    nom = prep.named_transformers_["nom"]
    nom_names = list(nom.get_feature_names_out(groups["NOM"]))
    num_names = list(groups["NUMERIC"])
    assert list(prep.get_feature_names_out()) == nom_names + num_names
    nom_set, num_pos = set(groups["NOM"]), {c: i for i, c in enumerate(num_names)}
    col_groups, claimed = [], np.zeros(len(nom_names) + len(num_names), int)
    for col in X.columns:
        if col in nom_set:
            idxs = [i for i, nm in enumerate(nom_names)
                    if nm.startswith(col + "_")]
        else:
            idxs = [len(nom_names) + num_pos[col]]
        assert idxs, f"no transformed columns for {col}"
        for i in idxs:
            assert claimed[i] == 0, f"column {i} claimed twice ({col})"
            claimed[i] = 1
        col_groups.append(idxs)
    assert claimed.sum() == len(claimed), "unclaimed transformed columns"
    assert len(col_groups) == X.shape[1] == 76
    return col_groups


def _aggregate(sv: np.ndarray, col_groups) -> np.ndarray:
    """Sum one-hot SHAP contributions back to the 76 original columns."""
    return np.column_stack([sv[:, idxs].sum(axis=1) for idxs in col_groups])


def cmd_shap() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from scipy import stats

    P7_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_xy()

    # final-model proxy: tuned LGBM (EXP-014a params) fit ONCE on ALL data.
    best = json.loads(BEST_PARAMS_JSON.read_text(encoding="utf-8"))
    params = best["lgbm"]["best_params"]
    t0 = time.perf_counter()
    pipe = build_tuned_lgbm_pipe()
    pipe.fit(X, y)
    fit_s = time.perf_counter() - t0
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    print(f"[p7] final-model proxy fit on all {len(X):,} rows "
          f"({fit_s / 60:.1f} min); params: {json.dumps(params, sort_keys=True)}")

    # stratified random sample of SHAP_SAMPLE_N rows (random_state=42)
    all_idx = np.arange(len(X))
    _, sample_idx = train_test_split(all_idx, test_size=SHAP_SAMPLE_N,
                                     stratify=y, random_state=RANDOM_STATE)
    sample_idx = np.sort(np.asarray(sample_idx))
    print(f"[p7] SHAP sample: {len(sample_idx):,} stratified rows "
          "(random_state=42)")

    explainer = shap.TreeExplainer(model)   # default path_dependent
    perturbation = explainer.feature_perturbation
    base = _base_value(explainer)
    print(f"[p7] TreeExplainer feature_perturbation={perturbation}; "
          f"base value (log-odds) = {base:.4f}")

    Xt_sample = prep.transform(X.iloc[sample_idx])
    sv = _shap_matrix(explainer, Xt_sample)
    col_groups = _original_space_groups(prep, X)
    agg = _aggregate(sv, col_groups)                     # (20k, 76)

    # additivity in original space: base + sum(agg) == raw margin score
    raw = model.predict_proba(Xt_sample, raw_score=True)
    add_dev = float(np.max(np.abs(base + agg.sum(axis=1) - raw)))
    print(f"[p7] additivity check: max |base + sum(SHAP) - raw_score| = "
          f"{add_dev:.2e}")
    if add_dev > 1e-3:
        raise SystemExit(f"[p7] SHAP additivity violated ({add_dev:.2e})")

    # global importance in ORIGINAL feature space
    imp_global = pd.DataFrame({
        "column": X.columns,
        "mean_abs_shap": np.abs(agg).mean(axis=0),
        "mean_shap": agg.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False,
                   kind="stable").reset_index(drop=True)
    imp_global.insert(0, "rank", np.arange(1, len(imp_global) + 1))
    p5 = pd.read_csv(P5_LGBM_IMPORTANCE_CSV)[["rank", "column", "mean_gain",
                                              "gain_share_pct"]]
    imp_global = imp_global.merge(
        p5.rename(columns={"rank": "p5_gain_rank",
                           "mean_gain": "p5_mean_gain"}),
        on="column", how="left")
    assert imp_global["p5_gain_rank"].notna().all()
    imp_global.to_csv(SHAP_IMPORTANCE_CSV, index=False, encoding="utf-8")
    print(f"[p7] wrote {SHAP_IMPORTANCE_CSV} (top 5: "
          + ", ".join(f"{r.column} ({r.mean_abs_shap:.4f})"
                      for r in imp_global.head(5).itertuples()) + ")")

    # ---- figure: beeswarm (top 20 original features) ----
    exp_obj = shap.Explanation(
        values=agg, base_values=np.full(len(agg), base),
        data=X.iloc[sample_idx].to_numpy(dtype=np.float32),
        feature_names=list(X.columns))
    shap.plots.beeswarm(exp_obj, max_display=BEESWARM_MAX_DISPLAY, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 9.0)
    fig.savefig(FIG_SHAP_SUMMARY, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[p7] wrote {FIG_SHAP_SUMMARY}")

    # ---- figure: mean|SHAP| rank vs P5 gain-importance rank ----
    rho, pval = stats.spearmanr(imp_global["rank"],
                                imp_global["p5_gain_rank"])
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.scatter(imp_global["p5_gain_rank"], imp_global["rank"], s=34,
               alpha=0.75, edgecolor="black", linewidth=0.4, color="#1f77b4")
    lim = (0, 78)
    ax.plot(lim, lim, linestyle="--", linewidth=0.9, color="#888888",
            label="perfect rank agreement")
    label_mask = (imp_global["rank"] <= 8) | (imp_global["p5_gain_rank"] <= 8)
    for r in imp_global[label_mask].itertuples():
        ax.annotate(r.column, (r.p5_gain_rank, r.rank),
                    xytext=(4, 3), textcoords="offset points", fontsize=7.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.invert_yaxis()   # rank 1 = most important at top-left
    ax.invert_xaxis()
    ax.set_xlabel("P5 LightGBM gain-importance rank (untuned EXP-006, "
                  "fold-averaged)")
    ax.set_ylabel("P7 mean |SHAP| rank (tuned EXP-014a, full-data proxy)")
    ax.set_title(f"Global importance rank agreement: SHAP vs split gain\n"
                 f"Spearman rho = {rho:.3f} (p = {pval:.2g}, n = 76 "
                 "original features)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_SHAP_VS_GAIN, dpi=150)
    plt.close("all")
    print(f"[p7] wrote {FIG_SHAP_VS_GAIN} (Spearman rho={rho:.4f}, "
          f"p={pval:.3g})")

    # ---- waterfalls for the 3 selected cases + shap_cases.parquet ----
    cases = json.loads(CASES_JSON.read_text(encoding="utf-8"))["cases"]
    case_rows = []
    for cid in ("A", "B", "C"):
        idx = int(cases[cid]["row_index"])
        Xt_case = prep.transform(X.iloc[[idx]])
        sv_case = _shap_matrix(explainer, Xt_case)
        agg_case = _aggregate(sv_case, col_groups)[0]
        raw_case = float(model.predict_proba(Xt_case, raw_score=True)[0])
        dev_case = float(abs(base + agg_case.sum() - raw_case))
        if dev_case > 1e-3:
            raise SystemExit(f"[p7] case {cid} additivity violated "
                             f"({dev_case:.2e})")
        exp_case = shap.Explanation(
            values=agg_case, base_values=base,
            data=X.iloc[idx].to_numpy(dtype=np.float32),
            feature_names=list(X.columns))
        shap.plots.waterfall(exp_case, max_display=WATERFALL_MAX_DISPLAY,
                             show=False)
        fig = plt.gcf()
        fig.set_size_inches(10.0, 9.0)
        fig.suptitle(
            f"Case {cid} - tuned LGBM (EXP-014a full-data proxy), original "
            f"feature space\ny={cases[cid]['y_true']} (GENHLTH="
            f"{cases[cid]['genhlth_raw']:.0f}), P(y=1) OOF="
            f"{cases[cid]['oof_lgbm']:.3f} - log-odds scale",
            fontsize=10, y=0.995)
        fig.savefig(FIGURES / f"p7_case_{cid}_waterfall.png", dpi=150,
                    bbox_inches="tight")
        plt.close("all")
        case_rows.append(pd.Series(agg_case, index=X.columns, name=cid))
        print(f"[p7] wrote {FIGURES / f'p7_case_{cid}_waterfall.png'} "
              f"(row {idx}, additivity dev {dev_case:.2e})")

    shap_cases = pd.concat(case_rows, axis=1).T
    shap_cases.index.name = "case"
    shap_cases["_base_value"] = base
    shap_cases["_row_index"] = [int(cases[c]["row_index"])
                                for c in ("A", "B", "C")]
    shap_cases.to_parquet(SHAP_CASES_PARQUET)
    print(f"[p7] wrote {SHAP_CASES_PARQUET}")

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": "tuned LGBMClassifier (EXP-014a params) refit on ALL "
                 "437,532 rows - FINAL-MODEL PROXY; its graded CV estimate "
                 "remains EXP-014a (0.904321) / EXP-015 nested (0.904303)",
        "params": params,
        "explainer": {
            "type": "shap.TreeExplainer",
            "feature_perturbation": str(perturbation),
            "rationale": ("default path_dependent chosen for report speed "
                          "(no background dataset needed); interventional "
                          "would be slower - recorded as a known quirk: "
                          "path-dependent can spread credit across "
                          "correlated one-hot dummies, which is why all "
                          "contributions are summed back to the 76 original "
                          "columns"),
            "base_value_log_odds": base,
            "output_space": "raw margin (log-odds)",
        },
        "sample": {"n": SHAP_SAMPLE_N, "stratified_on": "y",
                   "random_state": RANDOM_STATE},
        "additivity_max_abs_dev_sample": add_dev,
        "aggregation": "one-hot dummy-group SHAP values summed per original "
                       "column (structural map asserted to partition all "
                       f"transformed columns); beeswarm top "
                       f"{BEESWARM_MAX_DISPLAY}",
        "shap_vs_gain_spearman": {"rho": float(rho), "p_value": float(pval),
                                  "n_features": 76},
        "wall_s": {"proxy_fit": round(fit_s, 1)},
    }
    (P7_DIR / "shap_meta.json").write_text(json.dumps(meta, indent=2),
                                           encoding="utf-8")
    print(f"[p7] wrote {P7_DIR / 'shap_meta.json'}")


# ---- Part 4: summary ---------------------------------------------------------------
def cmd_summary() -> None:
    tax = pd.read_csv(TAXONOMY_CSV)
    # genhlth_class round-trips as strings from the mixed-type CSV column
    tax["genhlth_class"] = tax["genhlth_class"].astype(str)
    cand = pd.read_csv(CANDIDATES_CSV)
    cases = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    prov = json.loads(PROVENANCE_JSON.read_text(encoding="utf-8"))
    meta = json.loads((P7_DIR / "shap_meta.json").read_text(encoding="utf-8"))
    shap_imp = pd.read_csv(SHAP_IMPORTANCE_CSV)
    timing = json.loads(TIMING_JSON.read_text(encoding="utf-8")) \
        if TIMING_JSON.exists() else []

    ov = tax[tax["scope"] == "overall"].iloc[0]
    byc = tax[tax["scope"] == "by_genhlth_class"].set_index("genhlth_class")
    wk = tax[tax["scope"] == "wrong_in_k_models"].set_index("genhlth_class")

    n_shared = int(wk.loc["3", "n"])
    rho = meta["shap_vs_gain_spearman"]["rho"]

    def class_lines():
        out = []
        for c in ("1", "2", "3"):
            r = byc.loc[c]
            out.append(f"| {c} (negative: excellent/good) | {int(r['n']):,} "
                       f"| FP | {int(r['FP']):,} | {r['fp_rate']:.4f} |")
        for c in ("4", "5"):
            r = byc.loc[c]
            out.append(f"| {c} (positive: fair/poor) | {int(r['n']):,} "
                       f"| FN | {int(r['FN']):,} | {r['fn_rate']:.4f} |")
        return out

    case_lines = []
    for cid in ("A", "B", "C"):
        c = cases["cases"][cid]
        case_lines.append(
            f"- **Case {cid}** (row {c['row_index']}, fold {c['fold']}): "
            f"{c['rule']}. y={c['y_true']} (GENHLTH={c['genhlth_raw']:.0f}); "
            f"OOF P(y=1): LGBM {c['oof_lgbm']:.4f}, XGB {c['oof_xgb']:.4f}, "
            f"LR {c['oof_lr']:.4f}; wrong in "
            f"{c['n_models_wrong']}/3 models.")

    lines = [
        "# P7 — Error analysis + SHAP explainability (final model = tuned "
        "LightGBM, EXP-014a)",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} by "
        "`src/p7_error_xai.py`. **No `experiments_log.csv` rows were "
        "appended**: every number below derives from the OOF predictions of "
        "the already-logged configurations EXP-014a / EXP-014b / EXP-002a "
        "(deterministic consequence — see `results/p7/oof_provenance.json`) "
        "and from a single full-data refit of the EXP-014a model used as the "
        "final-model proxy for SHAP. Its graded CV estimate remains EXP-014a "
        "(mean CV AUC 0.904321; nested EXP-015 0.904303).",
        "",
        "Folds: the canonical `StratifiedKFold(10, shuffle=True, "
        "random_state=42)` via `src.config.make_folds()` (folds_sha256 "
        f"`{prov['folds_sha256']}`"
        "); OOF per-fold AUCs re-computed from the saved predictions match "
        "the logged fold scores to ≤ "
        + f"{max(v['max_abs_per_fold_auc_deviation'] for v in prov['verification'].values()):.1e}"
        " (max abs deviation, all three models).",
        "",
        "## Error taxonomy (tuned LGBM, OOF, threshold 0.5)",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| n (rows) | {int(ov['n']):,} |",
        f"| TP / FP / TN / FN | {int(ov['TP']):,} / {int(ov['FP']):,} / "
        f"{int(ov['TN']):,} / {int(ov['FN']):,} |",
        f"| FP rate (FPR = FP/N₀) | {ov['fp_rate']:.4f} |",
        f"| FN rate (FNR = FN/N₁) | {ov['fn_rate']:.4f} |",
        f"| TPR / TNR | {ov['tpr']:.4f} / {ov['tnr']:.4f} |",
        f"| error rate | {ov['error_rate']:.4f} |",
        "",
        "### Errors by raw GENHLTH class — the 3/4 boundary story",
        "",
        "| GENHLTH class | n | error type | count | rate |",
        "|---|---|---|---|---|",
        *class_lines(),
        "",
        f"- FP rate climbs monotonically with worse self-rated health "
        f"within the negatives: "
        + " → ".join(f"class {c}: {byc.loc[c, 'fp_rate']:.4f}"
                     for c in ("1", "2", "3"))
        + " (of true negatives, class 3 is misflagged most often).",
        f"- FN rate falls as health worsens: "
        + " → ".join(f"class {c}: {byc.loc[c, 'fn_rate']:.4f}"
                     for c in ("4", "5"))
        + " — class 4 (\"fair\") is the hardest positive to catch.",
        f"- Combined: {int(byc.loc['3', 'FP']) + int(byc.loc['4', 'FN']):,} "
        f"of {int(ov['FP']) + int(ov['FN']):,} errors "
        f"({100.0 * (int(byc.loc['3', 'FP']) + int(byc.loc['4', 'FN'])) / (int(ov['FP']) + int(ov['FN'])):.1f}%) "
        "sit exactly at the 3/4 boundary — the model's confusion is "
        "boundary ambiguity, not random noise.",
        "",
        "## Cross-model wrongness (LGBM ∪ XGB ∪ LR, OOF @ 0.5)",
        "",
        "| wrong in k/3 models | n | share |",
        "|---|---|---|",
        *[f"| {k} | {int(wk.loc[str(k), 'n']):,} | {wk.loc[str(k), 'prevalence']:.4f} |"
          for k in (0, 1, 2, 3)],
        "",
        f"- **{n_shared:,} instances are wrong in ALL 3 models** — the "
        "systematically hard residue (see `error_candidates.csv`, top "
        f"{len(cand)} by |oof_lgbm − 0.5|, most confident shared errors "
        "first).",
        f"- Shared-error label mix: "
        + ", ".join(
            f"genhlth {int(g)}: {int((cand['genhlth_raw'] == g).sum())}"
            for g in sorted(cand["genhlth_raw"].unique()))
        + f" (of the top {len(cand)} candidates).",
        "",
        "## Selected cases (full vectors + z-scores in selected_cases.json)",
        "",
        *case_lines,
        "",
        "Selection rationale: A/B isolate the model's most confident "
        "failures in each direction (top of the FP/FN confidence ordering); "
        "C isolates genuine 3/4-boundary ambiguity (wrong everywhere, "
        "probability at the fence).",
        "",
        "## SHAP (TreeExplainer, path-dependent) on the final-model proxy",
        "",
        f"- Explainer: `shap.TreeExplainer`, "
        f"`feature_perturbation={meta['explainer']['feature_perturbation']}` "
        "(default; chosen for report speed — no background pass needed; "
        "known quirk: path-dependent can spread credit across correlated "
        "one-hot dummies, which is exactly why all contributions are summed "
        "back to the 76 ORIGINAL columns before plotting).",
        f"- Output space: raw margin (log-odds); base value "
        f"{meta['explainer']['base_value_log_odds']:.4f}. Additivity "
        "asserted (max |base + ΣSHAP − raw_score| = "
        f"{meta['additivity_max_abs_dev_sample']:.1e} on the 20k sample and "
        "re-asserted per case).",
        f"- Global sample: {meta['sample']['n']:,} rows, stratified on y, "
        "random_state=42.",
        f"- Top-5 by mean|SHAP|: "
        + ", ".join(f"{r.column} ({r.mean_abs_shap:.4f})"
                    for r in shap_imp.head(5).itertuples()) + ".",
        f"- **SHAP-vs-gain rank agreement: Spearman ρ = {rho:.3f} "
        f"(p = {meta['shap_vs_gain_spearman']['p_value']:.2g}, n = 76)** — "
        "the tuned model's SHAP ordering broadly agrees with the untuned "
        "EXP-006 split-gain ordering (`figures/p7_shap_vs_gain.png`); "
        "divergences (rank gaps in the scatter) are typically "
        "collinear/split-credit features whose gain concentrates in one "
        "tree path but whose SHAP credit is shared.",
        "",
        "## Artifacts",
        "",
        "- `data/interim/oof_predictions.parquet` (oof_lgbm, oof_xgb, "
        "oof_lr, y_true, fold) + `results/p7/oof_provenance.json`",
        "- `results/p7/error_taxonomy.csv`, `error_candidates.csv`, "
        "`selected_cases.json`, `shap_global_importance.csv`, "
        "`shap_cases.parquet`, `shap_meta.json`",
        "- `figures/p7_shap_summary.png` (beeswarm, top 20 original "
        "features), `figures/p7_shap_vs_gain.png`, "
        "`figures/p7_case_{A,B,C}_waterfall.png` (all dpi 150)",
        "- Runtimes (results/p7/timing.json): "
        + ("; ".join(f"{t['cmd']} {t['wall_s'] / 60:.1f} min"
                     for t in timing) if timing else "n/a"),
        "",
    ]
    P7_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[p7] wrote {P7_SUMMARY_MD}")


# ---- CLI ---------------------------------------------------------------------------
COMMANDS = ("oof", "errors", "shap", "summary")
_FUNCS = {"oof": cmd_oof, "errors": cmd_errors, "shap": cmd_shap,
          "summary": cmd_summary}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="P7a error analysis + SHAP XAI on the final model "
                    "(tuned LGBM, EXP-014a); no experiments_log rows")
    ap.add_argument("targets", nargs="+", choices=list(COMMANDS) + ["all"])
    args = ap.parse_args(argv)
    order = COMMANDS if "all" in args.targets \
        else [c for c in COMMANDS if c in args.targets]
    for cmd in order:
        t0 = time.perf_counter()
        print(f"\n[p7] ===== {cmd} =====")
        _FUNCS[cmd]()
        _record_timing(cmd, time.perf_counter() - t0)
    print("\n[p7] done")


if __name__ == "__main__":
    main()
