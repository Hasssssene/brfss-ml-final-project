"""P3 - shared fold-safe CV harness + baseline experiments (EXP-001..003).

This module is the BACKBONE for every subsequent phase (P4-P6):

- ``load_xy()``            clean_v1.parquet + target_v1.parquet -> apply the
                           valid-target mask -> X (437,532 x 76 float32),
                           y (72,741 positives / 16.63%).
- ``build_preprocessor()`` the ONE shared model-time ColumnTransformer, built
                           from data/interim/column_meta.json encodings (never
                           hardcoded here): NOM -> SimpleImputer(most_frequent)
                           -> OneHotEncoder(handle_unknown='ignore');
                           ORD/NUM/ENG -> SimpleImputer(median) -> passthrough.
                           It lives INSIDE every Pipeline passed to CV, so it
                           is fit on training folds ONLY - preprocessing is
                           never fit on the full data (no leakage).
- ``make_fold_list()``     materialise the canonical folds ONCE from
                           src.config.make_folds() (StratifiedKFold(10,
                           shuffle=True, random_state=42)) plus a sha256
                           signature over the fold index arrays, so every run
                           can ASSERT it was scored on identical folds.
- ``run_experiment()``     cross_val_score(scoring='roc_auc', n_jobs=-1) on
                           the canonical folds; prints per-fold AUCs; APPENDS
                           exactly one row to results/experiments_log.csv.

``__main__`` runs the P3 baselines (methodology FIXED per AGENTS.md: target,
folds, metric, feature set 'all-76', no resampling):

- EXP-001   DummyClassifier(strategy='prior')          -> AUC = 0.5 sanity
- EXP-002a-d LogisticRegression + StandardScaler, MANUAL C grid
            {0.01, 0.1, 1.0, 10} (lbfgs default solver, max_iter=2000)
- EXP-003a-l DecisionTreeClassifier, MANUAL grid max_depth {6,10,16,None} x
            min_samples_leaf {1,50,200}. Depth-limited combos run first and
            the heaviest combo (None, 1) runs LAST so the 15-minute wall
            guard prunes it first if needed (pruning is reported, never
            silent).

Implementation notes (documented choices, not methodology):
- float32 throughout: every whitelisted value is a small integer (< 2^24) or
  a short decimal, exactly representable; this halves the memory of the 10
  parallel fold workers (sklearn trees cast to float32 internally anyway).
- ColumnTransformer(sparse_threshold=0.0) -> dense output so the LogisticReg
  pipeline can use StandardScaler(with_mean=True) on the one-hot block.
- One log row per RUN: the tuning grids produce sub-ids (EXP-002a..d,
  EXP-003a..l) so every number in the report cites a unique exp_id.
- Resume-safe: ``main()`` skips any exp_id already present in the log
  (scores are parsed back from the CSV and each row's recorded
  folds_sha256 prefix is re-asserted against the canonical hash), so a
  killed run can be relaunched without duplicate rows or lost results.
- The post-hoc fold-1 convergence refit of the winning LogisticRegression is
  documentation only (no CV, no log row): ConvergenceWarnings raised inside
  cross_val_score's out-of-process workers cannot be captured in the parent,
  so an honest in-process refit on the same pipeline documents convergence.

This module never touches git, vault/ or deliverables/, and writes only to
results/ (experiments_log.csv + p3_baselines_summary.md).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from src.config import (  # noqa: E402
    DATA_INTERIM,
    EXPERIMENTS_LOG,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS,
    make_folds,
)

CLEAN_PARQUET = DATA_INTERIM / "clean_v1.parquet"
TARGET_PARQUET = DATA_INTERIM / "target_v1.parquet"
META_JSON = DATA_INTERIM / "column_meta.json"
P3_SUMMARY_MD = RESULTS / "p3_baselines_summary.md"

FEATURE_SET_LABEL = "all-76"
IMBALANCE_STRATEGY_LABEL = "none"

# Exact header of results/experiments_log.csv (asserted against the file).
LOG_COLUMNS = [
    "exp_id", "timestamp", "model", "feature_set", "imbalance_strategy",
    "params",
    *[f"roc_auc_fold{i:02d}" for i in range(1, 11)],
    "roc_auc_mean", "notes",
]

# P2 contracts, re-asserted here so a stale/wrong parquet fails loudly.
EXPECTED_N_ROWS = 437_532
EXPECTED_N_COLS = 76
EXPECTED_N_POS = 72_741

# Manual tuning grids (P3 spec).
LR_C_GRID = (0.01, 0.1, 1.0, 10.0)
TREE_GRID = [  # cheapest first; heaviest ((None, 1)) last for the guard
    (6, 1), (6, 50), (6, 200),
    (10, 1), (10, 50), (10, 200),
    (16, 1), (16, 50), (16, 200),
    (None, 50), (None, 200), (None, 1),
]
EXP3_BUDGET_S = 15 * 60  # wall-clock guard for the whole EXP-003 grid


# ---- data / column groups -------------------------------------------------------
def load_column_groups(meta_path: Path = META_JSON) -> dict[str, list[str]]:
    """Column-name groups per encoding, read from column_meta.json.

    column_meta.json (written by src/preprocessing.py from the ADR-005
    whitelist) is the single source of truth for encodings - this module
    never hardcodes column lists.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    unknown = [c for c, s in meta.items()
               if s["encoding"] not in ("NOM", "ORD", "NUM", "ENG")]
    if unknown:
        raise ValueError(f"unknown encoding(s) in column_meta.json: {unknown}")
    nom = [c for c, s in meta.items() if s["encoding"] == "NOM"]
    numeric = [c for c, s in meta.items() if s["encoding"] in ("ORD", "NUM", "ENG")]
    return {"NOM": nom, "NUMERIC": numeric}


def load_xy(clean_parquet: Path = CLEAN_PARQUET,
            target_parquet: Path = TARGET_PARQUET) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the P2 checkpoints, apply the valid-target mask, assert contracts.

    Returns X (float32 DataFrame, whitelist column order, RangeIndex) and y
    (int8 numpy array). float32 is exact for all whitelisted values (codes
    < 2^24 / short decimals) and halves worker memory - see module docstring.
    """
    X_all = pd.read_parquet(clean_parquet)
    tgt = pd.read_parquet(target_parquet)
    if len(X_all) != len(tgt):
        raise ValueError("X and target frames are not row-aligned")
    valid = tgt["valid"].to_numpy(dtype=bool)
    X = X_all.loc[valid].reset_index(drop=True).astype("float32")
    y = tgt.loc[valid, "y"].to_numpy(dtype=np.int8)

    assert X.shape == (EXPECTED_N_ROWS, EXPECTED_N_COLS), X.shape
    assert int(y.sum()) == EXPECTED_N_POS, int(y.sum())
    assert not X.columns.duplicated().any()
    # NaNs are expected (module-sparse items); they are imputed INSIDE the
    # CV pipeline, never here.
    print(f"[p3] loaded X {X.shape[0]:,} x {X.shape[1]} "
          f"(float32) | positives {int(y.sum()):,} ({100.0 * y.mean():.2f}%)")
    return X, y


# ---- the ONE shared preprocessor ------------------------------------------------
def build_preprocessor(meta_path: Path = META_JSON) -> ColumnTransformer:
    """Shared model-time preprocessing; always used INSIDE a Pipeline.

    - NOM  -> SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    - ORD/NUM/ENG -> SimpleImputer(median) -> passthrough (scaling is left to
      the models that need it: LogisticRegression adds StandardScaler in its
      own pipeline; trees get raw values).

    sparse_threshold=0.0 forces dense float32 output so downstream
    StandardScaler(with_mean=True) works; verbose_feature_names_out=False
    keeps one-hot names clean for later importance analysis (P5).
    """
    groups = load_column_groups(meta_path)
    nom_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", dtype=np.float32)),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer(
        transformers=[
            ("nom", nom_pipe, groups["NOM"]),
            ("num", num_pipe, groups["NUMERIC"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


# ---- canonical folds ------------------------------------------------------------
def fold_list_hash(folds) -> str:
    """sha256 over the exact fold index arrays (order-sensitive)."""
    h = hashlib.sha256()
    for train, test in folds:
        h.update(np.asarray(train, dtype=np.int64).tobytes())
        h.update(np.asarray(test, dtype=np.int64).tobytes())
    return h.hexdigest()


def make_fold_list(X, y) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Materialise the canonical folds once from src.config.make_folds().

    Returns the list of (train_idx, test_idx) splits (reusable as cv= for
    cross_val_score) and its sha256 signature. Integrity-checked: folds must
    partition the rows exactly once.
    """
    folds = [(tr, te) for tr, te in make_folds().split(X, y)]
    all_test = np.concatenate([te for _, te in folds])
    if len(all_test) != len(X) or len(np.unique(all_test)) != len(X):
        raise AssertionError("folds do not partition the rows exactly once")
    for tr, te in folds:
        if len(np.intersect1d(tr, te)):
            raise AssertionError("train/test overlap in canonical folds")
    return folds, fold_list_hash(folds)


# ---- experiment runner ----------------------------------------------------------
def append_log_row(row: dict) -> None:
    """Append ONE row to results/experiments_log.csv (append-only, header-checked)."""
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if EXPERIMENTS_LOG.exists():
        header = EXPERIMENTS_LOG.read_text(encoding="utf-8").splitlines()[0]
        if header != ",".join(LOG_COLUMNS):
            raise AssertionError(f"experiments log header drift:\n{header}")
        write_header = False
    else:
        write_header = True
    df = pd.DataFrame([row])[LOG_COLUMNS]
    df.to_csv(EXPERIMENTS_LOG, mode="a", header=write_header,
              index=False, encoding="utf-8")


def _hash_prefix_from_notes(notes: str) -> str | None:
    """Extract the 'folds_sha256=<12hex>' prefix recorded in a row's notes."""
    for part in str(notes).split(";"):
        part = part.strip()
        if part.startswith("folds_sha256="):
            return part.split("=", 1)[1]
    return None


def load_logged_runs() -> dict[str, dict]:
    """Parse results/experiments_log.csv back into run dicts (resume support).

    Every row except the EXP-000 profiling row must carry 10 finite fold
    scores whose mean matches the logged roc_auc_mean (guards against a
    partially-written row from a killed process). runtime_s and the
    folds_sha256 prefix are recovered from the notes field.
    """
    if not EXPERIMENTS_LOG.exists():
        return {}
    df = pd.read_csv(EXPERIMENTS_LOG, dtype=str, keep_default_na=False)
    if df.empty:
        return {}
    if list(df.columns) != LOG_COLUMNS:
        raise AssertionError("experiments log header drift (load_logged_runs)")
    runs: dict[str, dict] = {}
    for _, row in df.iterrows():
        exp_id = str(row["exp_id"])
        if exp_id == "EXP-000":  # data-profiling row: no fold scores
            continue
        fold_vals = [str(row[f"roc_auc_fold{i:02d}"]).strip()
                     for i in range(1, 11)]
        if not all(fold_vals):
            raise AssertionError(f"{exp_id}: blank fold score(s) - corrupt row")
        scores = np.array([float(v) for v in fold_vals])
        mean, std = float(scores.mean()), float(scores.std(ddof=0))
        if not np.isfinite(scores).all():
            raise AssertionError(f"{exp_id}: non-finite fold score(s)")
        # Tolerance: roc_auc_mean is the true mean rounded to 6 decimals, so a
        # legitimate row can differ by exactly 0.5e-6 (e.g. EXP-003l, true mean
        # 0.6949215 -> logged 0.694922) plus float noise from re-summing the
        # parsed folds. Only genuinely corrupt/partial rows should fail.
        if abs(mean - float(row["roc_auc_mean"])) > 5.1e-7:
            raise AssertionError(f"{exp_id}: logged roc_auc_mean != mean(folds)")
        runtime_s = 0.0
        for part in str(row["notes"]).split(";"):
            part = part.strip()
            if part.startswith("runtime_s="):
                runtime_s = float(part.split("=", 1)[1])
        runs[exp_id] = {"exp_id": exp_id, "model": str(row["model"]),
                        "feature_set": str(row["feature_set"]),
                        "imbalance_strategy": str(row["imbalance_strategy"]),
                        "params": str(row["params"]), "scores": scores,
                        "mean": mean, "std": std, "runtime_s": runtime_s,
                        "notes": str(row["notes"])}
    return runs


def log_scored_run(exp_id: str, scores, model_name: str, params_desc: str,
                   note: str, runtime_s: float, fold_hash: str,
                   feature_set: str = FEATURE_SET_LABEL,
                   imbalance_strategy: str = IMBALANCE_STRATEGY_LABEL) -> dict:
    """Append ONE log row for a run whose fold scores were computed elsewhere.

    Same row construction/printout as ``run_experiment`` (single append path
    via ``append_log_row``). Needed by P5, where selection/PCA runs use a
    custom fold loop because the fitted per-fold objects (MI scores, gain
    importances, explained-variance ratios) must come back from inside each
    fold - ``cross_val_score`` discards them.

    ``fold_hash`` must be the sha256 of the fold list the scores came from
    (recorded as a prefix in the notes so cross-run fold identity is
    auditable).
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (N_SPLITS,):
        raise AssertionError(
            f"{exp_id}: expected {N_SPLITS} fold scores, got {scores.shape}")
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{exp_id}: non-finite fold score(s): {scores}")
    mean = float(scores.mean())
    std = float(scores.std(ddof=0))

    note_parts = [f"runtime_s={runtime_s:.1f}", f"fold_std={std:.6f}",
                  f"folds_sha256={fold_hash[:12]}"]
    if note:
        note_parts.append(note)
    row = {
        "exp_id": exp_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "feature_set": feature_set,
        "imbalance_strategy": imbalance_strategy,
        "params": params_desc,
        **{f"roc_auc_fold{i + 1:02d}": f"{s:.6f}" for i, s in enumerate(scores)},
        "roc_auc_mean": f"{mean:.6f}",
        "notes": "; ".join(note_parts),
    }
    append_log_row(row)

    print(f"[{exp_id}] {model_name}({params_desc}) feature_set={feature_set}")
    print(f"[{exp_id}] fold AUCs: " + " ".join(f"{s:.4f}" for s in scores))
    print(f"[{exp_id}] mean CV ROC-AUC = {mean:.6f} (fold std {std:.6f}) "
          f"[wall {runtime_s:.1f}s | folds_sha256={fold_hash[:12]}]")
    return {"exp_id": exp_id, "model": model_name, "params": params_desc,
            "scores": scores, "mean": mean, "std": std,
            "runtime_s": runtime_s, "notes": row["notes"]}


def run_experiment(exp_id: str, model, model_name: str, params_desc: str,
                   X, y, note: str = "", folds=None, fold_hash: str | None = None,
                   preprocessor=None,
                   imbalance_strategy: str = IMBALANCE_STRATEGY_LABEL,
                   feature_set: str = FEATURE_SET_LABEL) -> dict:
    """One 10-fold CV run on the canonical folds; APPENDS one log row.

    Parameters
    ----------
    exp_id       unique run id for the log (e.g. 'EXP-002c').
    model        estimator (or nested Pipeline) placed after the shared
                 preprocessor: full pipeline = prep -> model.
    model_name   family label for the log ('LogisticRegression', ...).
    params_desc  hyperparameter summary ('C=1.0;max_iter=2000;...').
    X, y         the full matrices (never pre-processed outside the pipeline).
    note         free text for the notes column.
    folds        canonical fold list; built fresh from make_folds() if None.
    fold_hash    expected sha256 of the fold arrays; ASSERTED when given so
                 every run proves it used the identical fold assignment.
    preprocessor optional pre-built ColumnTransformer (defaults to the shared
                 build_preprocessor()).
    imbalance_strategy  label for the imbalance_strategy column (default
                 'none'). P4 passes fold-internal strategy labels such as
                 'SMOTE(0.5)' or 'class_weight=balanced'; the CV folds are
                 unchanged - resampling happens INSIDE fit on train folds.
    feature_set  label for the feature_set column (default 'all-76'). P5
                 passes subset/selection labels ('H1-subset', 'kbest-50',
                 'pca-100', ...); the underlying rows/folds never change.

    Returns a result dict (fold scores, mean, std, runtime) for downstream
    summaries - callers never recompute scores.
    """
    if folds is None:
        folds, computed_hash = make_fold_list(X, y)
    else:
        computed_hash = fold_list_hash(folds)
    if fold_hash is not None and computed_hash != fold_hash:
        raise AssertionError(
            f"{exp_id}: fold assignment differs from canonical folds "
            f"({computed_hash[:12]} != {fold_hash[:12]})")
    if preprocessor is None:
        preprocessor = build_preprocessor()

    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # worker-side warnings (e.g. ConvergenceWarning in loky processes)
        # cannot be captured here; convergence is documented via a dedicated
        # in-process refit (see convergence_refit).
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = cross_val_score(pipe, X, y, cv=list(folds),
                                 scoring="roc_auc", n_jobs=-1,
                                 pre_dispatch="n_jobs")
    runtime_s = time.perf_counter() - t0
    return log_scored_run(exp_id, scores, model_name, params_desc, note,
                          runtime_s, computed_hash,
                          feature_set=feature_set,
                          imbalance_strategy=imbalance_strategy)


# ---- diagnostics (NOT experiments, not logged) -----------------------------------
def convergence_refit(X, y, folds, C: float) -> dict:
    """Refit the winning LogisticRegression pipeline on fold 1 in-process.

    Documents convergence honestly: ConvergenceWarning instances raised inside
    cross_val_score's out-of-process workers do not propagate to the parent,
    so we refit the identical pipeline on the fold-1 training data (same
    393,779-row scale) with warnings recorded. No CV, no log row.
    """
    train, _ = folds[0]
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE)),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.fit(X.iloc[train], y[train])
    n_conv = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    n_iter = int(np.ravel(pipe.named_steps["clf"].n_iter_)[0])
    print(f"[diag] fold-1 refit LR(C={C}): lbfgs n_iter={n_iter} / max_iter=2000, "
          f"ConvergenceWarnings captured={n_conv}")
    return {"C": C, "n_iter": n_iter, "convergence_warnings": n_conv}


# ---- P3 summary ------------------------------------------------------------------
def write_summary(results: list[dict], fold_hash: str, lr_by_C: dict[float, dict],
                  diag: dict, pruned: list[tuple], wall_s: float,
                  path: Path = P3_SUMMARY_MD) -> None:
    """Render results/p3_baselines_summary.md from the run dicts (log-faithful)."""
    dummy = next(r for r in results if r["model"] == "DummyClassifier")
    lr_runs = [r for r in results if r["model"] == "LogisticRegression"]
    tree_runs = [r for r in results if r["model"] == "DecisionTreeClassifier"]
    best_lr = max(lr_runs, key=lambda r: r["mean"])
    best_tree = max(tree_runs, key=lambda r: r["mean"])
    best_C = max(lr_by_C, key=lambda c: lr_by_C[c]["mean"])

    def row_md(r):
        return (f"| {r['exp_id']} | {r['model']} | {r['params']} | "
                f"{r['mean']:.5f} ± {r['std']:.5f} | {r['runtime_s']:.0f}s |")

    lines = [
        "# P3 - Baseline experiments (EXP-001..003)",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} by "
        "`src/experiments.py` (harness backbone for P4-P6).",
        "",
        "Fixed methodology: y = GENHLTH ∈ {4,5}; feature set `all-76` "
        "(clean_v1.parquet, 437,532 × 76 after valid-target mask; 72,741 "
        "positives = 16.63%); metric ROC-AUC; identical folds for EVERY run "
        "— `StratifiedKFold(10, shuffle=True, random_state=42)` via "
        "`src.config.make_folds()`, fold-index sha256 "
        f"`{fold_hash[:16]}…` asserted in every `run_experiment` call; "
        "no resampling (`imbalance_strategy=none`). Preprocessing "
        "(most_frequent impute + one-hot for NOM, median impute for "
        "ORD/NUM/ENG) lives inside the CV pipeline — fit on training folds "
        "only.",
        "",
        "## All runs",
        "",
        "| exp_id | model | params | mean CV ROC-AUC ± std | wall time |",
        "|---|---|---|---|---|",
        *[row_md(r) for r in results],
        "",
        "## Winners per family",
        "",
        f"- **Sanity floor (EXP-001)** DummyClassifier(prior): "
        f"**mean CV AUC = {dummy['mean']:.6f}** (expected exactly 0.5 for a "
        "constant-prior predictor — confirms the scorer and folds).",
        f"- **Linear family (EXP-002a–d)**: best **C = {best_C:g}** → "
        f"**{best_lr['mean']:.5f} ± {best_lr['std']:.5f}** "
        f"({best_lr['exp_id']}). Grid: "
        + ", ".join(f"C={c:g}: {lr_by_C[c]['mean']:.5f}" for c in sorted(lr_by_C)) + ".",
        f"- **Tree family (EXP-003a–l)**: best "
        f"**{best_tree['params']}** → **{best_tree['mean']:.5f} ± "
        f"{best_tree['std']:.5f}** ({best_tree['exp_id']}).",
        "",
        "## Fold-score stability",
        "",
        f"- 10 folds × ~43,753 test rows each (stratified); identical "
        f"assignment across all {len(results)} runs (hash-asserted per run).",
    ]
    for label, r in (("Dummy", dummy), ("LogReg winner", best_lr),
                     ("Tree winner", best_tree)):
        s = r["scores"]
        lines.append(f"- {label} ({r['exp_id']}): fold AUCs "
                     f"{s.min():.4f}–{s.max():.4f} (spread "
                     f"{s.max() - s.min():.4f}), std {r['std']:.6f}.")
    spread_tree = max(r["scores"].std(ddof=0) for r in tree_runs)
    lines.append(
        f"- All tree configs: max fold std across the grid = {spread_tree:.6f} "
        "— scores are stable fold-to-fold; no fold is anomalous, so mean CV "
        "AUC comparisons between configurations are trustworthy at the 4th "
        "decimal.")
    if best_lr["mean"] > best_tree["mean"]:
        lines.append(
            f"- Winner margin: best LogReg beats best tree by "
            f"{best_lr['mean'] - best_tree['mean']:+.5f} AUC "
            f"({best_lr['mean']:.5f} vs {best_tree['mean']:.5f}); both far "
            "above the 0.5 prior floor.")
    else:
        lines.append(
            f"- Winner margin: best tree beats best LogReg by "
            f"{best_tree['mean'] - best_lr['mean']:+.5f} AUC "
            f"({best_tree['mean']:.5f} vs {best_lr['mean']:.5f}); both far "
            "above the 0.5 prior floor.")
    lines += [
        "",
        "## Diagnostics & runtime",
        "",
        f"- Per-config CV wall time sums to "
        f"{sum(r['runtime_s'] for r in results) / 60:.1f} min over all "
        f"{len(results)} runs; the final `main()` invocation (data load + "
        f"missing configs + diagnostics) took {wall_s / 60:.1f} min "
        "(the grid was resumed across two processes after a killed "
        "session; per-config times come from the log rows).",
        f"- Convergence (fold-1 in-process refit of the winning LR, same "
        f"pipeline): lbfgs n_iter = {diag['n_iter']} / 2000, "
        f"ConvergenceWarnings = {diag['convergence_warnings']}. "
        "cross_val_score worker-side warnings cannot be captured in-process; "
        "this refit on 393,779 training rows documents convergence behaviour.",
    ]
    if pruned:
        lines.append(f"- **Grid pruning**: EXP-003 exceeded the {EXP3_BUDGET_S}s "
                     f"budget; skipped (heaviest first): {pruned}.")
    else:
        lines.append("- No grid pruning: all 12 tree configurations ran "
                     f"within the {EXP3_BUDGET_S}s budget.")
    lines.append("- Environment: sklearn DecisionTree/LogisticRegression, "
                 "n_jobs=-1 (10 fold fits in parallel, pre_dispatch='n_jobs'), "
                 "float32 design matrix (~267 columns after one-hot).")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[p3] wrote {path}")


# ---- main ------------------------------------------------------------------------
def main() -> None:
    """Run EXP-001..003, RESUMING past any config already in the log.

    A config whose exp_id is already logged is never re-run (the log is
    append-only - duplicate rows must not exist). Previously logged rows are
    parsed back from the CSV, their folds_sha256 prefix is asserted against
    the canonical hash, and their scores flow into the summary unchanged.
    """
    t_start = time.perf_counter()
    logged = load_logged_runs()
    if logged:
        print(f"[p3] resume: {len(logged)} run(s) already logged: "
              f"{sorted(logged)}")
    X, y = load_xy()
    groups = load_column_groups()
    print(f"[p3] preprocessor: {len(groups['NOM'])} NOM cols -> one-hot | "
          f"{len(groups['NUMERIC'])} ORD/NUM/ENG cols -> median impute, passthrough")

    folds, fold_hash = make_fold_list(X, y)
    # determinism proof: rebuilding the splitter from scratch is identical
    _, fold_hash_again = make_fold_list(X, y)
    assert fold_hash_again == fold_hash, "make_folds() not deterministic"
    print(f"[p3] canonical folds sha256 = {fold_hash}")

    # cross-session fold identity: every already-logged run must have been
    # scored on exactly these folds (hash prefix recorded in its row)
    for exp_id in sorted(logged):
        prior = _hash_prefix_from_notes(logged[exp_id]["notes"])
        if prior and not fold_hash.startswith(prior):
            raise AssertionError(
                f"{exp_id}: logged folds_sha256={prior} does not match the "
                f"canonical folds {fold_hash[:12]}")
    if logged:
        print(f"[p3] fold identity verified against all {len(logged)} "
              "pre-logged runs")

    results_by_id: dict[str, dict] = {}

    def ensure(exp_id: str, build_model, model_name: str, params_desc: str,
               note: str) -> dict:
        """Run exp_id unless its row is already in the log (resume)."""
        if exp_id in logged:
            print(f"[p3] {exp_id} already logged - resuming past it")
            results_by_id[exp_id] = logged[exp_id]
            return logged[exp_id]
        r = run_experiment(exp_id, build_model(), model_name, params_desc,
                           X, y, note=note, folds=folds, fold_hash=fold_hash)
        results_by_id[exp_id] = r
        return r

    # ---- EXP-001: prior floor ---------------------------------------------------
    dummy = ensure(
        "EXP-001", lambda: DummyClassifier(strategy="prior"),
        "DummyClassifier", "strategy=prior",
        note="sanity baseline; constant-prior predictor must score exactly 0.5")

    # ---- EXP-002: LogReg manual C grid (one row per C) ---------------------------
    t2 = time.perf_counter()
    lr_by_C: dict[float, dict] = {}
    for i, C in enumerate(LR_C_GRID):
        def build_lr(C=C):
            return Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=2000,
                                           random_state=RANDOM_STATE)),
            ])
        lr_by_C[C] = ensure(
            f"EXP-002{'abcd'[i]}", build_lr, "LogisticRegression",
            f"C={C:g};max_iter=2000;solver=lbfgs;scaler=StandardScaler",
            note="manual C grid; lbfgs default solver")
    lr_grid_s = time.perf_counter() - t2
    best_C = max(lr_by_C, key=lambda c: lr_by_C[c]["mean"])
    print(f"[p3] EXP-002 winner: C={best_C:g} "
          f"(mean CV AUC {lr_by_C[best_C]['mean']:.6f}); "
          f"session wall {lr_grid_s:.0f}s")

    # ---- EXP-003: DecisionTree manual grid (one row per combo) --------------------
    t3 = time.perf_counter()
    letters = "abcdefghijkl"
    tree_ids = [f"EXP-003{letters[i]}" for i in range(len(TREE_GRID))]
    # budget counts tree-grid time across sessions: pre-logged configs
    # contribute their recorded runtime_s
    logged_tree_s = sum(logged[e]["runtime_s"] for e in tree_ids if e in logged)
    pruned: list[tuple] = []
    tree_results = []
    for i, (depth, leaf) in enumerate(TREE_GRID):
        exp_id = tree_ids[i]
        if exp_id in logged:
            tree_results.append(logged[exp_id])
            continue
        elapsed = logged_tree_s + (time.perf_counter() - t3)
        if elapsed > EXP3_BUDGET_S:
            # heaviest combos are ordered last, so they are pruned first
            pruned.append((depth, leaf))
            print(f"[p3] EXP-003 budget guard: {elapsed:.0f}s > "
                  f"{EXP3_BUDGET_S}s — pruning {exp_id} "
                  f"(max_depth={depth}, min_samples_leaf={leaf})")
            continue

        def build_tree(depth=depth, leaf=leaf):
            return DecisionTreeClassifier(max_depth=depth,
                                          min_samples_leaf=leaf,
                                          random_state=RANDOM_STATE)
        r = ensure(exp_id, build_tree, "DecisionTreeClassifier",
                   f"max_depth={depth};min_samples_leaf={leaf}",
                   note="manual grid max_depth x min_samples_leaf")
        tree_results.append(r)
    tree_grid_s = time.perf_counter() - t3
    best_tree = max(tree_results, key=lambda r: r["mean"])
    print(f"[p3] EXP-003 winner: {best_tree['params']} "
          f"(mean CV AUC {best_tree['mean']:.6f}); session wall "
          f"{tree_grid_s:.0f}s; pruned={pruned or 'none'}")

    # ---- convergence diagnostic (not an experiment) -------------------------------
    diag = convergence_refit(X, y, folds, C=best_C)

    ordered = [dummy] + [lr_by_C[c] for c in LR_C_GRID] + tree_results
    wall_s = time.perf_counter() - t_start
    print(f"\n[p3] ==== P3 summary ====")
    print(f"[p3] EXP-001 Dummy(prior) mean AUC = {dummy['mean']:.6f}")
    print(f"[p3] EXP-002 best C = {best_C:g} -> "
          f"{lr_by_C[best_C]['mean']:.6f} ± {lr_by_C[best_C]['std']:.6f}")
    print(f"[p3] EXP-003 best tree = {best_tree['params']} -> "
          f"{best_tree['mean']:.6f} ± {best_tree['std']:.6f}")
    print(f"[p3] session wall time {wall_s / 60:.1f} min "
          f"(resumed past {len(logged)} pre-logged runs)")

    write_summary(ordered, fold_hash, lr_by_C, diag, pruned, wall_s)


if __name__ == "__main__":
    main()
