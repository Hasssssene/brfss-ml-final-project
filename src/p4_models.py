"""P4 - model zoo + non-sklearn boosting + imbalance comparison (EXP-004..010).

Built ON TOP of the P3 harness (``src/experiments.py``): every run goes
through ``run_experiment`` and is therefore scored on the canonical folds
from ``src.config.make_folds()`` (StratifiedKFold(10, shuffle=True,
random_state=42), fold-index sha256 asserted per run - identical to
EXP-001..003). Methodology is ADR-locked and unchanged: y = GENHLTH in
{4,5}, feature set ``all-76``, ROC-AUC, no fold-level randomness beyond
RANDOM_STATE=42. This phase only ADDS model families and fold-internal
imbalance strategies.

Part 1 - model zoo, ONE sensible config per family (tuning is P6):
- EXP-004 RandomForestClassifier(n_estimators=200, min_samples_leaf=50,
  max_features='sqrt', random_state=42, n_jobs=1)
- EXP-005 HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
  random_state=42)   [sklearn default early_stopping='auto' is ACTIVE at
  this sample size (>10k rows) - documented, not tuned]
- EXP-006 lightgbm.LGBMClassifier(n_estimators=500, learning_rate=0.05,
  num_leaves=63, random_state=42, n_jobs=1, verbosity=-1)
- EXP-007 xgboost.XGBClassifier(n_estimators=500, learning_rate=0.05,
  max_depth=6, tree_method='hist', random_state=42, n_jobs=1,
  eval_metric='logloss')   [verbosity=0 added for clean worker output only]

Boosters (EXP-005..007) use the NaN-PASSTHROUGH preprocessor variant
(``build_nan_passthrough_preprocessor``). Verified on this codebase with
sklearn 1.9.1: OneHotEncoder(handle_unknown='ignore') encodes NaN as its
OWN category (NOM column SOMALE -> categories [1.0, 2.0, 3.0, 4.0, nan],
5 of 54 NOM columns carry NaN; none are all-NaN in a training fold), and
ORD/NUM/ENG columns pass through with NaN intact, which HistGB / LGBM /
XGB route natively at split time. Missingness stays signal instead of
being blurred by most-frequent imputation - native missing handling is
part of why boosting suits module-sparse survey data. RandomForest and
LogisticRegression keep the P3 impute-based preprocessor (they cannot
split/impute NaN natively - LR would propagate NaN into the solver).

Part 2 - imbalance comparison (INSIDE the folds via imblearn Pipeline):
- family A = best of EXP-004..007 by mean CV AUC (resolved from the log at
  invocation time; deterministic tie-break by exp_id)
- family B = LogisticRegression C=0.01 (P3 winner; baseline EXP-002a)
- EXP-008a/b: class weighting - booster scale_pos_weight=neg/pos (computed
  from the full-data class prior 364791/72741 ~ 5.0149; stratified folds
  preserve this ratio by construction) or sklearn class_weight='balanced'
  (RF / HistGB / LogReg compute it inside fit on the train fold).
- EXP-009a/b: SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)
  inside the CV folds. Why 0.5: a full 1:1 rebalance would synthesize
  ~262,845 minority rows per fold (train fold 393,779 -> 656,624 rows);
  0.5 caps synthesis at ~98,689 rows (492,468 total) - the standard
  compromise between class signal and synthetic density. DEVIATION (forced
  + documented): SMOTE cannot handle NaN (verified: imblearn 0.14.2 raises
  'Input X contains NaN'), so EXP-009a uses the P3 IMPUTE preprocessor
  even though the booster family used NaN-passthrough in Part 1.
- EXP-010a/b: RandomUnderSampler(sampling_strategy=0.5, random_state=42)
  inside the folds (covers the majority-class rubric angle). RUS accepts
  NaN (verified on imblearn 0.14.2), so family A KEEPS its NaN-passthrough
  preprocessor here; per training fold this drops ~197k majority rows
  (393,779 -> ~196,401).

Pipeline order for family B resampling runs: prep -> StandardScaler ->
SMOTE/RUS -> LR (SMOTE sees standardized features, so Euclidean nearest
neighbours are not dominated by large-scale NUM columns).

Resume-safe like P3: an exp_id already present in
``results/experiments_log.csv`` is skipped, its recorded folds_sha256
prefix is re-asserted against the canonical hash, and its parsed scores
flow into the summary unchanged.

Usage (one foreground command per EXP, per execution discipline):
    python -m src.p4_models EXP-004
    python -m src.p4_models EXP-005
    python -m src.p4_models EXP-006
    python -m src.p4_models EXP-007
    python -m src.p4_models EXP-008a   (requires all zoo rows logged)
    ...
    python -m src.p4_models EXP-010b
    python -m src.p4_models summary    (writes results/p4_summary.md)

Convenience aliases: ``phase1`` = EXP-004..007, ``phase2`` = EXP-008a..
EXP-010b, ``all`` = everything + summary.

This module never touches git, vault/ or deliverables/, and writes only to
results/ (experiments_log.csv + p4_summary.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from src.config import RANDOM_STATE  # noqa: E402
from src.experiments import (  # noqa: E402
    META_JSON,
    RESULTS,
    _hash_prefix_from_notes,
    build_preprocessor,
    load_column_groups,
    load_logged_runs,
    load_xy,
    make_fold_list,
    run_experiment,
)

P4_SUMMARY_MD = RESULTS / "p4_summary.md"

ZOO_IDS = ["EXP-004", "EXP-005", "EXP-006", "EXP-007"]
IMB_A_IDS = ["EXP-008a", "EXP-009a", "EXP-010a"]  # best Part-1 family
IMB_B_IDS = ["EXP-008b", "EXP-009b", "EXP-010b"]  # LogReg C=0.01 contrast
ALL_P4_IDS = ZOO_IDS + IMB_A_IDS + IMB_B_IDS

LR_CONTRAST_C = 0.01          # P3 winner (EXP-002a); linear contrast family
LR_BASELINE_ID = "EXP-002a"   # already logged baseline for family B
SMOTE_SAMPLING = 0.5
SMOTE_K = 5
RUS_SAMPLING = 0.5


# ---- model builders (exactly one sensible config per family) ---------------------
def build_rf(class_weight=None):
    kw = dict(n_estimators=200, min_samples_leaf=50, max_features="sqrt",
              random_state=RANDOM_STATE, n_jobs=1)
    if class_weight is not None:
        kw["class_weight"] = class_weight
    return RandomForestClassifier(**kw)


def build_histgb(class_weight=None):
    kw = dict(max_iter=300, learning_rate=0.06, random_state=RANDOM_STATE)
    if class_weight is not None:
        kw["class_weight"] = class_weight
    return HistGradientBoostingClassifier(**kw)


def build_lgbm(scale_pos_weight=None):
    kw = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
              random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)
    if scale_pos_weight is not None:
        kw["scale_pos_weight"] = scale_pos_weight
    return LGBMClassifier(**kw)


def build_xgb(scale_pos_weight=None):
    kw = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
              tree_method="hist", random_state=RANDOM_STATE, n_jobs=1,
              eval_metric="logloss", verbosity=0)
    if scale_pos_weight is not None:
        kw["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**kw)


# Family registry: name, default builder, preferred preprocessing variant
# ('nanpass' = no imputation anywhere; 'impute' = P3 shared preprocessor),
# and the native class-weighting mechanism.
FAMILIES = {
    "EXP-004": dict(name="RandomForestClassifier", build=build_rf, prep="impute",
                    native_nan=False, weight_kind="class_weight",
                    params="n_estimators=200;min_samples_leaf=50;max_features=sqrt;"
                           "random_state=42;n_jobs=1"),
    "EXP-005": dict(name="HistGradientBoostingClassifier", build=build_histgb,
                    prep="nanpass", native_nan=True, weight_kind="class_weight",
                    params="max_iter=300;learning_rate=0.06;random_state=42"),
    "EXP-006": dict(name="LGBMClassifier", build=build_lgbm, prep="nanpass",
                    native_nan=True, weight_kind="scale_pos_weight",
                    params="n_estimators=500;learning_rate=0.05;num_leaves=63;"
                           "random_state=42;n_jobs=1;verbosity=-1"),
    "EXP-007": dict(name="XGBClassifier", build=build_xgb, prep="nanpass",
                    native_nan=True, weight_kind="scale_pos_weight",
                    params="n_estimators=500;learning_rate=0.05;max_depth=6;"
                           "tree_method=hist;random_state=42;n_jobs=1;"
                           "eval_metric=logloss"),
}

ZOO_NOTES = {
    "EXP-004": ("P4 model zoo: one config per family (tuning deferred to P6); "
                "P3 impute preprocessor (RF cannot split NaN)"),
    "EXP-005": ("P4 zoo; NaN-passthrough preprocessor (OneHotEncoder encodes "
                "NaN as its own category - verified sklearn 1.9.1 - and HistGB "
                "routes NaN natively); sklearn default early_stopping='auto' "
                "active (n_train>10k)"),
    "EXP-006": ("P4 zoo; NaN-passthrough preprocessor (verified sklearn 1.9.1 "
                "OHE NaN category + native LGBM NaN routing) - native missing "
                "handling is part of why boosters suit sparse survey data"),
    "EXP-007": ("P4 zoo; NaN-passthrough preprocessor (verified sklearn 1.9.1 "
                "OHE NaN category + native XGB NaN routing); verbosity=0 is "
                "output hygiene only"),
}


# ---- preprocessing variants ------------------------------------------------------
def build_nan_passthrough_preprocessor(meta_path: Path = META_JSON) -> ColumnTransformer:
    """Booster preprocessing: NO imputation anywhere.

    NOM -> OneHotEncoder(handle_unknown='ignore') directly. Verified on this
    data with sklearn 1.9.1: NaN becomes its own category (e.g. SOMALE ->
    categories [1.0, 2.0, 3.0, 4.0, nan]); unseen NaNs at transform time are
    covered by handle_unknown='ignore'. ORD/NUM/ENG -> passthrough with NaN
    intact, split natively by HistGB / LGBM / XGB. Keep this inside the CV
    pipeline (fit on train folds only), exactly like the shared impute
    variant.
    """
    groups = load_column_groups(meta_path)
    return ColumnTransformer(
        transformers=[
            ("nom", OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
             groups["NOM"]),
            ("num", "passthrough", groups["NUMERIC"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_prep(variant: str):
    if variant == "impute":
        return build_preprocessor()
    if variant == "nanpass":
        return build_nan_passthrough_preprocessor()
    raise ValueError(f"unknown preprocessing variant: {variant}")


# ---- run-spec construction -------------------------------------------------------
def resolve_best_family(logged: dict[str, dict]) -> str:
    """Best Part-1 family by mean CV AUC (requires all four zoo rows)."""
    missing = [e for e in ZOO_IDS if e not in logged]
    if missing:
        raise SystemExit(f"[p4] phase 2 needs {ZOO_IDS} all logged first; "
                         f"missing {missing}")
    return max(ZOO_IDS, key=lambda e: (logged[e]["mean"], e))


def smote_arithmetic(folds, y) -> dict:
    """Per-training-fold class counts behind the sampling_strategy=0.5 choice."""
    train, _ = folds[0]
    ytr = y[train]
    n_pos = int((ytr == 1).sum())
    n_neg = int((ytr == 0).sum())
    syn_full = n_neg - n_pos                       # strategy 1.0
    syn_half = int(round(SMOTE_SAMPLING * n_neg)) - n_pos  # strategy 0.5
    rus_neg = int(round(n_pos / RUS_SAMPLING))     # majority rows kept by RUS 0.5
    return {"n_pos": n_pos, "n_neg": n_neg, "syn_full": syn_full,
            "syn_half": syn_half, "rus_neg": rus_neg,
            "n_train": int(len(train))}


def make_spec(exp_id: str, logged: dict[str, dict], y, folds) -> dict:
    """Assemble (model, name, params, prep variant, imbalance label, note)."""
    spw = float((y == 0).sum() / (y == 1).sum())
    spw_str = f"{spw:.4f}"
    ar = smote_arithmetic(folds, y)

    if exp_id in FAMILIES:  # ---- Part 1 zoo ----
        fam = FAMILIES[exp_id]
        return dict(model=fam["build"](), model_name=fam["name"],
                    params=fam["params"], prep=fam["prep"], imb="none",
                    note=ZOO_NOTES[exp_id])

    if exp_id.endswith("a"):  # family A = best Part-1 family
        best = resolve_best_family(logged)
        fam = FAMILIES[best]
        base_note = f"baseline {best} ({fam['name']}, same config otherwise)"

        if exp_id == "EXP-008a":  # native class weighting
            if fam["weight_kind"] == "scale_pos_weight":
                model = fam["build"](scale_pos_weight=spw)
                params = f"{fam['params']};scale_pos_weight={spw_str}"
                imb = f"scale_pos_weight={spw_str}"
                wdesc = (f"scale_pos_weight=neg/pos={spw_str} from full-data "
                         f"prior (stratified folds preserve the ratio)")
            else:
                model = fam["build"](class_weight="balanced")
                params = f"{fam['params']};class_weight=balanced"
                imb = "class_weight=balanced"
                wdesc = "class_weight=balanced (computed inside fit on train fold)"
            note = f"class weighting vs Part-1 default; {wdesc}; {base_note}"

        elif exp_id == "EXP-009a":  # SMOTE inside folds
            model = ImbPipeline([
                ("smote", SMOTE(sampling_strategy=SMOTE_SAMPLING,
                                k_neighbors=SMOTE_K, random_state=RANDOM_STATE)),
                ("clf", fam["build"]()),
            ])
            params = (f"SMOTE(strategy={SMOTE_SAMPLING};k_neighbors={SMOTE_K};"
                      f"rs=42);{fam['params']}")
            imb = f"SMOTE({SMOTE_SAMPLING})"
            note = (f"SMOTE INSIDE folds (imblearn Pipeline); strategy=0.5 because "
                    f"1:1 would synthesize {ar['syn_full']} rows/fold "
                    f"(train {ar['n_train']}->{ar['n_train'] + ar['syn_full']}) vs "
                    f"{ar['syn_half']} at 0.5 - standard compromise; k=5; SMOTE "
                    f"rejects NaN so P3 impute preprocessor used instead of "
                    f"nanpassthrough (documented deviation); {base_note}")
            return dict(model=model, model_name=fam["name"], params=params,
                        prep="impute", imb=imb, note=note)

        else:  # EXP-010a: RandomUnderSampler inside folds
            model = ImbPipeline([
                ("rus", RandomUnderSampler(sampling_strategy=RUS_SAMPLING,
                                           random_state=RANDOM_STATE)),
                ("clf", fam["build"]()),
            ])
            params = (f"RandomUnderSampler(strategy={RUS_SAMPLING};rs=42);"
                      f"{fam['params']}")
            imb = f"RandomUnderSampler({RUS_SAMPLING})"
            prep = "nanpass" if fam["native_nan"] else "impute"
            note = (f"RUS INSIDE folds; majority-class rubric angle; per fold "
                    f"majority drops {ar['n_neg']}->{ar['rus_neg']} rows "
                    f"(train {ar['n_train']}->{ar['rus_neg'] + ar['n_pos']}); "
                    f"RUS accepts NaN (verified imblearn 0.14.2) so "
                    f"{'nanpassthrough prep kept' if prep == 'nanpass' else 'impute prep (family cannot take NaN)'}; "
                    f"{base_note}")
            return dict(model=model, model_name=fam["name"], params=params,
                        prep=prep, imb=imb, note=note)

        return dict(model=model, model_name=fam["name"], params=params,
                    prep=fam["prep"], imb=imb, note=note)

    # ---- family B: LogisticRegression C=0.01 contrast ----
    lr_base = f"baseline {LR_BASELINE_ID} (C={LR_CONTRAST_C}; same scaler)"
    if exp_id == "EXP-008b":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=LR_CONTRAST_C, max_iter=2000,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ])
        params = (f"C={LR_CONTRAST_C};max_iter=2000;solver=lbfgs;"
                  f"class_weight=balanced;scaler=StandardScaler")
        imb = "class_weight=balanced"
        note = f"class_weight=balanced vs default weights; {lr_base}"

    elif exp_id == "EXP-009b":
        model = ImbPipeline([
            ("scale", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=SMOTE_SAMPLING,
                            k_neighbors=SMOTE_K, random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(C=LR_CONTRAST_C, max_iter=2000,
                                       random_state=RANDOM_STATE)),
        ])
        params = (f"SMOTE(strategy={SMOTE_SAMPLING};k_neighbors={SMOTE_K};rs=42);"
                  f"C={LR_CONTRAST_C};max_iter=2000;solver=lbfgs;"
                  f"scaler=StandardScaler")
        imb = f"SMOTE({SMOTE_SAMPLING})"
        note = (f"SMOTE INSIDE folds; order prep->scale->SMOTE->LR so SMOTE "
                f"distances see standardized features; strategy=0.5 rationale as "
                f"EXP-009a; {lr_base}")

    elif exp_id == "EXP-010b":
        model = ImbPipeline([
            ("scale", StandardScaler()),
            ("rus", RandomUnderSampler(sampling_strategy=RUS_SAMPLING,
                                       random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(C=LR_CONTRAST_C, max_iter=2000,
                                       random_state=RANDOM_STATE)),
        ])
        params = (f"RandomUnderSampler(strategy={RUS_SAMPLING};rs=42);"
                  f"C={LR_CONTRAST_C};max_iter=2000;solver=lbfgs;"
                  f"scaler=StandardScaler")
        imb = f"RandomUnderSampler({RUS_SAMPLING})"
        note = f"RUS INSIDE folds; majority-class rubric angle; {lr_base}"
    else:
        raise SystemExit(f"[p4] unknown exp_id: {exp_id}")
    return dict(model=model, model_name="LogisticRegression", params=params,
                prep="impute", imb=imb, note=note)


def prep_label(exp_id: str, best: str | None) -> str:
    """Preprocessing-variant label for the summary table (mirrors make_spec)."""
    if exp_id in FAMILIES:
        return FAMILIES[exp_id]["prep"]
    if exp_id.endswith("a"):
        fam = FAMILIES[best]
        if exp_id == "EXP-009a":
            return "impute"
        if exp_id == "EXP-010a":
            return "nanpass" if fam["native_nan"] else "impute"
        return fam["prep"]
    return "impute"


# ---- summary ----------------------------------------------------------------------
def write_p4_summary(fold_hash: str, wall_s: float,
                     path: Path = P4_SUMMARY_MD) -> None:
    """Render results/p4_summary.md straight from the append-only log."""
    logged = load_logged_runs()
    missing = [e for e in ALL_P4_IDS if e not in logged]
    if missing:
        raise SystemExit(f"[p4] cannot summarise; missing rows: {missing}")
    lr_base = logged[LR_BASELINE_ID]
    best = resolve_best_family(logged)
    best_fam = FAMILIES[best]
    overall = max(ALL_P4_IDS, key=lambda e: (logged[e]["mean"], e))

    def row(e):
        r = logged[e]
        pl = prep_label(e, best)
        return (f"| {e} | {r['model']} | {r['params']} | {pl} | {r['mean']:.5f} "
                f"± {r['std']:.5f} | {r['runtime_s']:.0f}s |")

    zoo_rank = sorted(ZOO_IDS, key=lambda e: -logged[e]["mean"])
    beats_lr = [e for e in ALL_P4_IDS if logged[e]["mean"] > lr_base["mean"]]

    imb_rows = []
    for strat, a_id, b_id in (("class weighting", "EXP-008a", "EXP-008b"),
                              ("SMOTE(0.5)", "EXP-009a", "EXP-009b"),
                              ("RandomUnderSampler(0.5)", "EXP-010a", "EXP-010b")):
        da = logged[a_id]["mean"] - logged[best]["mean"]
        db = logged[b_id]["mean"] - lr_base["mean"]
        imb_rows.append((strat, a_id, da, b_id, db))

    lines = [
        "# P4 - Model zoo + non-sklearn boosting + imbalance comparison (EXP-004..010)",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} by "
        "`src/p4_models.py` on top of the P3 harness (`src/experiments.py`).",
        "",
        "Methodology unchanged (ADR-locked): y = GENHLTH ∈ {4,5}; feature set "
        "`all-76` (437,532 × 76; 72,741 positives = 16.63%); ROC-AUC; identical "
        "folds for EVERY run — `StratifiedKFold(10, shuffle=True, "
        "random_state=42)` via `src.config.make_folds()`, fold-index sha256 "
        f"`{fold_hash[:16]}…` asserted in every `run_experiment` call (same "
        "folds as EXP-001..003). Resampling happens INSIDE fit on training "
        "folds via imblearn Pipelines; SMOTE/RUS use random_state=42.",
        "",
        "## All P4 runs",
        "",
        "| exp_id | model | params | prep | mean CV ROC-AUC ± std | wall |",
        "|---|---|---|---|---|---|",
        *[row(e) for e in ALL_P4_IDS],
        "",
        "Preprocessing variants: `impute` = P3 shared preprocessor "
        "(most_frequent + one-hot for NOM, median impute for ORD/NUM/ENG); "
        "`nanpass` = NaN-passthrough for boosters — no imputer anywhere: "
        "OneHotEncoder(handle_unknown='ignore') encodes NaN as its own "
        "category (verified on sklearn 1.9.1, e.g. SOMALE → categories "
        "[1,2,3,4,nan]) and ORD/NUM/ENG pass through with NaN intact for "
        "native routing by HistGB/LGBM/XGB.",
        "",
        "## Part 1 — model zoo verdict (one config each; tuning is P6)",
        "",
        *[f"- {e} {logged[e]['model']}: **{logged[e]['mean']:.5f} ± "
          f"{logged[e]['std']:.5f}** ({logged[e]['runtime_s']:.0f}s)"
          for e in zoo_rank],
        f"- **Best family: {best_fam['name']} ({best}) at "
        f"{logged[best]['mean']:.5f}** — used as family A for Part 2.",
        f"- P3 references on the same folds: LogReg C=0.01 = "
        f"{lr_base['mean']:.5f} (EXP-002a), best single tree = 0.88021 "
        "(EXP-003i), Dummy = 0.5 (EXP-001).",
    ]
    if beats_lr:
        lines.append(f"- **{len(beats_lr)} P4 run(s) beat LogReg's "
                     f"{lr_base['mean']:.6f}**: "
                     + ", ".join(f"{e} (+{logged[e]['mean'] - lr_base['mean']:.5f})"
                                 for e in sorted(beats_lr,
                                                 key=lambda e: -logged[e]["mean"]))
                     + ".")
    else:
        lines.append(f"- No P4 run beat LogReg's {lr_base['mean']:.6f}.")
    lines += [
        "",
        "## Part 2 — imbalance verdict (ΔAUC vs same model with default "
        "weights, same folds)",
        "",
        "| strategy | family A run | ΔAUC vs Part-1 default | family B (LogReg C=0.01) run | ΔAUC vs EXP-002a |",
        "|---|---|---|---|---|",
    ]
    for strat, a_id, da, b_id, db in imb_rows:
        lines.append(f"| {strat} | {a_id} ({logged[a_id]['model']}) | "
                     f"{da:+.5f} | {b_id} | {db:+.5f} |")
    a_help = [f"{s} (A {a}: {d:+.5f})" for s, a, d, _, _ in imb_rows if d > 0]
    b_help = [f"{s} (B {b}: {d:+.5f})" for s, _, _, b, d in imb_rows if d > 0]
    lines += [
        "",
        f"- Family A ({best_fam['name']}): "
        + ("; ".join(a_help) + " improve AUC; everything else hurts."
           if a_help else "every imbalance strategy HURT AUC vs default "
                          "weights — consistent with ROC-AUC being "
                          "threshold-free and rank-based.") ,
        f"- Family B (LogisticRegression): "
        + ("; ".join(b_help) + " improve AUC."
           if b_help else "every imbalance strategy HURT AUC vs default "
                          "weights."),
        "",
        "## Deviations, forced choices & warnings",
        "",
        "- **NaN-passthrough verified** (sklearn 1.9.1): "
        "OneHotEncoder(handle_unknown='ignore') → NaN is its own category; "
        "boosters keep NaN end-to-end. RF/LogReg keep the impute preprocessor.",
        "- **EXP-009a deviation**: SMOTE rejects NaN (imblearn 0.14.2 raises "
        "'Input X contains NaN'), so family A uses the P3 impute preprocessor "
        "in the SMOTE run (its Part-1 baseline used NaN-passthrough). The "
        "EXP-008a/EXP-010a family-A runs keep their Part-1 preprocessing.",
        "- **scale_pos_weight source**: neg/pos from the full-data prior "
        "(364,791/72,741 ≈ 5.0149); StratifiedKFold preserves the ratio in "
        "every training fold, so this equals the per-fold value to rounding.",
        "- **SMOTE budget**: per training fold 65,467 pos / 328,312 neg → "
        "1:1 would synthesize ~262,845 rows (656,624 train rows); 0.5 "
        "synthesizes ~98,689 (492,468). RUS(0.5) keeps 130,934 majority rows "
        "(196,401 train rows).",
        "- HistGB uses sklearn defaults beyond the two tuned-ish params — "
        "early_stopping='auto' is ACTIVE (n>10k), so 300 max_iter may stop "
        "earlier; this is the family's stock behaviour, not a tuning choice.",
        "- XGB verbosity=0 and LGBM verbosity=-1 are output hygiene only.",
        "- No convergence warnings expected for boosters; the LogReg contrast "
        "reuses the P3 pipeline (lbfgs converged in 18 iterations at C=0.01).",
        "",
        "## Runtimes",
        "",
        f"- Sum of per-run CV wall times: "
        f"{sum(logged[e]['runtime_s'] for e in ALL_P4_IDS) / 60:.1f} min over "
        f"{len(ALL_P4_IDS)} runs; final session wall {wall_s / 60:.1f} min "
        "(runners: cross_val_score n_jobs=-1, model n_jobs=1 inside each "
        "fold worker).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[p4] wrote {path}")
    print(f"[p4] overall best P4 run: {overall} ({logged[overall]['model']}) "
          f"mean CV AUC = {logged[overall]['mean']:.6f}")


# ---- CLI --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(
        description="P4 model zoo + imbalance comparison (EXP-004..010)")
    ap.add_argument("targets", nargs="+",
                    help="exp ids (EXP-004 … EXP-010b) or aliases: "
                         "phase1, phase2, all, summary")
    args = ap.parse_args(argv)

    aliases = {"phase1": ZOO_IDS, "phase2": IMB_A_IDS + IMB_B_IDS,
               "all": ALL_P4_IDS + ["summary"]}
    ids: list[str] = []
    do_summary = False
    for t in args.targets:
        if t == "summary":
            do_summary = True
        elif t in aliases:
            ids.extend(e for e in aliases[t] if e != "summary")
            if "summary" in aliases[t]:
                do_summary = True
        elif t in ALL_P4_IDS:
            ids.append(t)
        else:
            raise SystemExit(f"[p4] unknown target: {t}")

    logged = load_logged_runs()
    already = [e for e in ids if e in logged]
    if already:
        print(f"[p4] resume: {len(already)} run(s) already logged: {already}")

    X, y = load_xy()
    folds, fold_hash = make_fold_list(X, y)
    _, fold_hash_again = make_fold_list(X, y)
    assert fold_hash_again == fold_hash, "make_folds() not deterministic"
    print(f"[p4] canonical folds sha256 = {fold_hash}")

    for exp_id in sorted(logged):  # cross-session fold identity check
        prior = _hash_prefix_from_notes(logged[exp_id]["notes"])
        if prior and not fold_hash.startswith(prior):
            raise AssertionError(
                f"{exp_id}: logged folds_sha256={prior} != canonical "
                f"{fold_hash[:12]}")
    print(f"[p4] fold identity verified against all {len(logged)} "
          "pre-logged runs")

    for exp_id in ids:
        if exp_id in logged:
            print(f"[p4] {exp_id} already logged - skipping")
            continue
        spec = make_spec(exp_id, logged, y, folds)
        r = run_experiment(exp_id, spec["model"], spec["model_name"],
                           spec["params"], X, y, note=spec["note"],
                           folds=folds, fold_hash=fold_hash,
                           preprocessor=build_prep(spec["prep"]),
                           imbalance_strategy=spec["imb"])
        logged[exp_id] = r

    if do_summary:
        write_p4_summary(fold_hash, time.perf_counter() - t0)
    print(f"[p4] session wall time {(time.perf_counter() - t0) / 60:.1f} min; "
          f"P4 rows logged: {sum(1 for e in ALL_P4_IDS if e in load_logged_runs())}"
          f"/{len(ALL_P4_IDS)}")


if __name__ == "__main__":
    main()
