#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BRFSS 2021 -- predict Fair/Poor general health (GENHLTH 4-5 vs 1-3).

End-to-end pipeline: load raw survey -> whitelist (303 -> 76 columns) ->
recode sentinels per codebook -> exploratory profile (target, missingness,
associations - report section 4) -> NaN-passthrough pipeline -> tuned
LightGBM (EXP-014a) -> stratified 10-fold CV: ROC-AUC + confusion matrix.

Every random component is seeded RANDOM_STATE = 42; the only CV splitter
is StratifiedKFold(10, shuffle=True, random_state=42).

Usage:
    python code.py --full    # real run (437,532 rows, ~2 min)
    python code.py --smoke   # fast check on a 50k stratified subsample

Expects the raw CSV at <repo>/data/raw/LLCP2021.csv (two levels up).
Final model traceability: EXP-014a in results/experiments_log.csv.
"""


from __future__ import annotations

# [1] CONFIG & CONSTANTS
import argparse
import hashlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import lightgbm

try:                                    # EDA figures only; stats print without it
    import matplotlib
    matplotlib.use("Agg")               # headless-safe: savefig only, no windows
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:                     # matplotlib absent -> stats-only EDA
    plt = None
    Rectangle = None

RANDOM_STATE = 42                       # the only seed used anywhere
N_SPLITS = 10
POSITIVE_CLASSES = (4, 5)               # y = 1 if GENHLTH in {4,5} (Fair/Poor)
NEGATIVE_CLASSES = (1, 2, 3)            # y = 0 if GENHLTH in {1,2,3}
TARGET_COL = "GENHLTH"

# paths (relative to this script: deliverables/../...)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_RAW = ROOT / "data" / "raw" / "LLCP2021.csv"
RESULTS_DIR = ROOT / "results"
SUMMARY_TXT = RESULTS_DIR / "final_run_summary.txt"
FOLD_SCORES_CSV = RESULTS_DIR / "final_run_fold_scores.csv"

# data contracts (asserted after cleaning; mirrors the P2 phase)
EXPECTED_RAW_ROWS = 438_693             # raw CSV rows
EXPECTED_N_FEATURES = 76                # ADR-005 whitelist size
EXPECTED_N_VALID = 437_532              # rows after dropping invalid target
EXPECTED_N_POS = 72_741                 # positives (16.63% class rate)
EXPECTED_STATE_NUNIQUE = 53             # 50 states + DC + PR + GU

# canonical fold identity
# sha256 over the canonical fold index arrays (same hash function as the
# project harness); first 16 hex chars recorded in every logged run.
CANONICAL_FOLDS_SHA16 = "3b3a17457039b258"

# the tuned final model (EXP-014a)
# Verbatim from results/p6/best_params.json ("lgbm" -> "best_params"):
# winner of RandomizedSearchCV(n_iter=15, inner 5-fold, roc_auc, rs=42).
TUNED_LGBM_PARAMS = dict(
    colsample_bytree=0.5979914312095727,
    learning_rate=0.021908197340907744,
    min_child_samples=44,
    n_estimators=512,
    num_leaves=94,
    reg_lambda=3.1234907221207773,
    subsample=0.815876852955632,
)
# Fixed booster flags, exactly as the harness constructed the model:
# subsample>0 requires subsample_freq>0 to activate bagging; n_jobs=1 keeps
# each fold's fit single-threaded (parallelism is across folds); verbosity
# silences LightGBM logging inside worker processes.
LGBM_FIXED_FLAGS = dict(
    random_state=RANDOM_STATE,
    n_jobs=1,
    verbosity=-1,
    subsample_freq=1,
)

# reference values for the verification check (full mode only)
# EXP-014a row of results/experiments_log.csv (values logged at 6 decimals).
EXP014A_FOLDS = (
    0.904011, 0.905465, 0.905302, 0.903872, 0.905757,
    0.901129, 0.903399, 0.902115, 0.906891, 0.905271,
)
EXP014A_MEAN = 0.904321
MATCH_TOL = 1e-6

SMOKE_N = 50_000                        # --smoke stratified subsample size


# [2] WHITELIST + RECODE SPECS  (ported verbatim from src/preprocessing.py)
# The ADR-005 column whitelist: the 76 kept columns of the 303 in the raw
# file, each with its encoding and per-column recode contract.
#   encoding: ORD (ordinal codes, natural order) | NOM (one-hot at train
#             time) | NUM (plain numeric) | ENG (engineered recode)
#   recode:   ("none",) | ("sent", sentinel_codes) | named engineered
#             recodes (daycnt / alc / fv / bmi / ...)
# Sentinel sets are codebook-exact: 7/77/777 = Don't know, 9/99/999 =
# Refused, 88 = "none" (day counts), 7777/9999 = height/weight style
# sentinels - and crucially NOT applied where those codes are legitimate
# values (e.g. EMPLOY1 7 = "Retired", INCOME3 7/9 = income brackets,
# _RACE 7 = "Multiracial", PRIMINSR 7/9 = TRICARE / state plan).
# The leakage blacklist (GENHLTH = the target source, _RFHLTH = its
# calculated fair/poor flag, _PHYS14D/_MENT14D = GENHLTH-family binned
# derivatives) is asserted absent from the matrix below.
BLACKLIST = ("GENHLTH", "_RFHLTH", "_PHYS14D", "_MENT14D")

_WHITELIST_ROWS = (
    # -- health days (Section 2) --
    ("PHYSHLTH", "ENG", ("daycnt",)),
    ("MENTHLTH", "ENG", ("daycnt",)),
    ("POORHLTH", "ENG", ("daycnt",)),
    # -- health care access (Section 3) --
    ("PRIMINSR", "NOM", ("sent", (77, 99))),       # 1-10 + 88 none; 7/9 valid plans
    ("PERSDOC3", "NOM", ("sent", (7, 9))),
    ("MEDCOST1", "NOM", ("sent", (7, 9))),
    ("CHECKUP1", "ORD", ("sent", (7, 9))),         # 8 = never kept as high value
    # -- exercise (Section 4) --
    ("EXERANY2", "NOM", ("sent", (7, 9))),
    # -- chronic conditions (Sections 5-7) --
    ("BPHIGH6", "NOM", ("sent", (7, 9))),
    ("BPMEDS", "NOM", ("sent", (7, 9))),
    ("CHOLCHK3", "NOM", ("sent", (7, 9))),         # rule 7: non-monotone code order
    ("TOLDHI3", "NOM", ("sent", (7, 9))),
    ("CHOLMED3", "NOM", ("sent", (7, 9))),
    ("CVDINFR4", "NOM", ("sent", (7, 9))),
    ("CVDCRHD4", "NOM", ("sent", (7, 9))),
    ("CVDSTRK3", "NOM", ("sent", (7, 9))),
    ("ASTHMA3", "NOM", ("sent", (7, 9))),
    ("CHCSCNCR", "NOM", ("sent", (7, 9))),
    ("CHCOCNCR", "NOM", ("sent", (7, 9))),
    ("CHCCOPD3", "NOM", ("sent", (7, 9))),
    ("ADDEPEV3", "NOM", ("sent", (7, 9))),
    ("CHCKDNY2", "NOM", ("sent", (7, 9))),
    ("DIABETE4", "NOM", ("sent", (7, 9))),
    # -- arthritis (Section 8) --
    ("HAVARTH5", "NOM", ("sent", (7, 9))),
    ("ARTHEXER", "NOM", ("sent", (7, 9))),
    ("ARTHEDU", "NOM", ("sent", (7, 9))),
    ("LMTJOIN3", "NOM", ("sent", (7, 9))),
    ("ARTHDIS2", "NOM", ("sent", (7, 9))),
    ("JOINPAI2", "NUM", ("sent", (77, 99))),       # 0-10 scale; 7/9 valid ratings
    # -- demographics (Section 9) --
    ("SEXVAR", "NOM", ("none",)),
    ("MARITAL", "NOM", ("sent", (9,))),
    ("EDUCA", "ORD", ("sent", (9,))),              # 1-6
    ("RENTHOM1", "NOM", ("sent", (7, 9))),
    ("VETERAN3", "NOM", ("sent", (7, 9))),
    ("EMPLOY1", "NOM", ("sent", (9,))),            # 7 = Retired is VALID
    ("CHILDREN", "ENG", ("child",)),               # 88->0, 99->NaN (no 77 code)
    ("INCOME3", "ORD", ("sent", (77, 99))),        # 1-11; 7/9 valid brackets
    # -- disability (Section 10) --
    ("DEAF", "NOM", ("sent", (7, 9))),
    ("BLIND", "NOM", ("sent", (7, 9))),
    ("DECIDE", "NOM", ("sent", (7, 9))),
    ("DIFFWALK", "NOM", ("sent", (7, 9))),
    ("DIFFDRES", "NOM", ("sent", (7, 9))),
    ("DIFFALON", "NOM", ("sent", (7, 9))),
    # -- tobacco (Section 11) --
    ("_SMOKER3", "ORD", ("sent", (9,))),           # 1-4
    ("USENOW3", "NOM", ("sent", (7, 9))),
    ("ECIGNOW1", "NOM", ("sent", (7, 9))),
    # -- alcohol (Section 12) --
    ("ALCDAY5", "ENG", ("alc",)),                  # decode to drinking days/month
    ("AVEDRNK3", "ENG", ("avedrnk",)),             # 88 none -> 0
    ("DRNK3GE5", "ENG", ("drnk",)),                # 88 none -> 0
    ("MAXDRNKS", "NUM", ("sent", (77, 99))),
    # -- immunization / HIV (Sections 13-14) --
    ("FLUSHOT7", "NOM", ("sent", (7, 9))),
    ("FLSHTMY3", "NOM", ("mmyyyy",)),              # 777777/999999 -> NaN
    ("IMFVPLA2", "NOM", ("sent", (77, 99))),       # 1-12; 7/9 valid places
    ("PNEUVAC4", "NOM", ("sent", (7, 9))),
    ("HIVTST7", "NOM", ("sent", (7, 9))),
    # -- fruits & vegetables (Section 15) --
    ("FRUIT2", "ENG", ("fv",)),
    ("FRUITJU2", "ENG", ("fv",)),
    ("FVGREEN1", "ENG", ("fv",)),
    ("FRENCHF1", "ENG", ("fv",)),
    ("POTATOE1", "ENG", ("fv",)),
    ("VEGETAB2", "ENG", ("fv",)),
    # -- module survivors (>= 25% effective coverage after sentinel recode) --
    ("PREDIAB1", "NOM", ("sent", (7, 9))),         # 26.0% coverage
    ("HOMBPCHK", "NOM", ("sent", (7, 9))),         # 28.5%
    ("HOMRGCHK", "NOM", ("sent", (7, 9))),         # 28.6%
    ("CAREGIV1", "NOM", ("sent", (7, 9))),         # 8 = recipient died, valid
    ("CRGVEXPT", "NOM", ("sent", (7, 9))),         # 31.4%
    ("MARIJAN1", "ENG", ("marj",)),                # 88->0
    ("SOMALE", "NOM", ("sent", (7, 9))),           # 25.2%
    ("SOFEMALE", "NOM", ("sent", (7, 9))),         # 28.8%
    ("TRNSGNDR", "NOM", ("sent", (7, 9))),         # 55.2%
    # -- admin kept --
    ("_STATE", "NOM", ("none",)),                  # 53 FIPS codes, no sentinels
    ("QSTLANG", "NOM", ("none",)),                 # judgment keep: interview language
    ("_URBSTAT", "NOM", ("none",)),
    # -- calculated keepers --
    ("_BMI5", "ENG", ("bmi",)),                    # /100, 2 implied decimals
    ("_RACE", "NOM", ("sent", (9,))),              # 1-8; 7 = Multiracial valid
    ("_AGE80", "NUM", ("none",)),                  # 18-80, imputed, no sentinels
)
assert len(_WHITELIST_ROWS) == EXPECTED_N_FEATURES, len(_WHITELIST_ROWS)
WHITELIST = {col: {"encoding": enc, "recode": recode}
             for col, enc, recode in _WHITELIST_ROWS}

# Post-recode validity contracts: every non-NaN value must be in the allowed
# set / range. Doubles as a residual-sentinel scan (a missed sentinel would
# violate its column's allowed set).
VALID_SETS = {
    "PRIMINSR": set(range(1, 11)) | {88},
    "PERSDOC3": {1, 2, 3},
    "MEDCOST1": {1, 2},
    "CHECKUP1": {1, 2, 3, 4, 8},
    "EXERANY2": {1, 2},
    "BPHIGH6": {1, 2, 3, 4},
    "BPMEDS": {1, 2},
    "CHOLCHK3": {1, 2, 3, 4, 5, 6, 8},
    "TOLDHI3": {1, 2},
    "CHOLMED3": {1, 2},
    "CVDINFR4": {1, 2}, "CVDCRHD4": {1, 2}, "CVDSTRK3": {1, 2},
    "ASTHMA3": {1, 2}, "CHCSCNCR": {1, 2}, "CHCOCNCR": {1, 2},
    "CHCCOPD3": {1, 2}, "ADDEPEV3": {1, 2}, "CHCKDNY2": {1, 2},
    "DIABETE4": {1, 2, 3, 4},
    "HAVARTH5": {1, 2}, "ARTHEXER": {1, 2}, "ARTHEDU": {1, 2},
    "LMTJOIN3": {1, 2}, "ARTHDIS2": {1, 2},
    "SEXVAR": {1, 2},
    "MARITAL": set(range(1, 7)),
    "EDUCA": set(range(1, 7)),
    "RENTHOM1": {1, 2, 3},
    "VETERAN3": {1, 2},
    "EMPLOY1": set(range(1, 9)),
    "INCOME3": set(range(1, 12)),
    "DEAF": {1, 2}, "BLIND": {1, 2}, "DECIDE": {1, 2}, "DIFFWALK": {1, 2},
    "DIFFDRES": {1, 2}, "DIFFALON": {1, 2},
    "_SMOKER3": {1, 2, 3, 4},
    "USENOW3": {1, 2, 3},
    "ECIGNOW1": {1, 2, 3, 4},
    "FLUSHOT7": {1, 2},
    "FLSHTMY3": {2020} | set(range(12020, 123022)) | set(range(772018, 772023)),
    "IMFVPLA2": set(range(1, 13)),
    "PNEUVAC4": {1, 2},
    "HIVTST7": {1, 2},
    "PREDIAB1": {1, 2, 3},
    "HOMBPCHK": {1, 2}, "HOMRGCHK": {1, 2},
    "CAREGIV1": {1, 2, 8},
    "CRGVEXPT": {1, 2},
    "SOMALE": {1, 2, 3, 4}, "SOFEMALE": {1, 2, 3, 4},
    "TRNSGNDR": {1, 2, 3, 4},
    "QSTLANG": {1, 2},
    "_URBSTAT": {1, 2},
    "_RACE": set(range(1, 9)),
    "_STATE": set(range(1, 79)),
}
VALID_RANGES = {
    "PHYSHLTH": (0, 30), "MENTHLTH": (0, 30), "POORHLTH": (0, 30),
    "CHILDREN": (0, 87),
    "ALCDAY5": (0, 31.0),
    "AVEDRNK3": (0, 76), "DRNK3GE5": (0, 76), "MAXDRNKS": (1, 76),
    "JOINPAI2": (0, 10),
    "MARIJAN1": (0, 30),
    "FRUIT2": (0, 3100), "FRUITJU2": (0, 3100), "FVGREEN1": (0, 3100),
    "FRENCHF1": (0, 3100), "POTATOE1": (0, 3100), "VEGETAB2": (0, 3100),
    "_BMI5": (10.0, 99.99),
    "_AGE80": (18, 80),
}

# recode implementations (verbatim port)
def _recode_sent(s: pd.Series, codes) -> pd.Series:
    return s.where(~s.isin(list(codes)))


def _recode_daycnt(s: pd.Series) -> pd.Series:
    """1-30 days, 88 = 'none' -> 0, 77/99 -> NaN (PHYSHLTH family)."""
    out = s.where(~s.isin([77, 99]))
    return out.replace({88: 0})


def _recode_drnk(s: pd.Series) -> pd.Series:
    """1-76 times, 88 = 'none' -> 0, 77/99 -> NaN (DRNK3GE5)."""
    out = s.where(~s.isin([77, 99]))
    return out.replace({88: 0})


def _recode_avedrnk(s: pd.Series) -> pd.Series:
    """1-76 avg drinks, 88 = 'none' -> 0, 77/99 -> NaN (AVEDRNK3)."""
    out = s.where(~s.isin([77, 99]))
    return out.replace({88: 0})


def _recode_child(s: pd.Series) -> pd.Series:
    """1-87 children, 88 = 'none' -> 0, 99 = refused -> NaN (no 77 code)."""
    out = s.where(~s.isin([99]))
    return out.replace({88: 0})


def _recode_marj(s: pd.Series) -> pd.Series:
    """1-30 days, 88 = 'none' -> 0, 77/99 -> NaN (MARIJAN1)."""
    out = s.where(~s.isin([77, 99]))
    return out.replace({88: 0})


def _recode_alc(s: pd.Series) -> pd.Series:
    """ALCDAY5 -> drinking days per month (float).

    101-107 days/week -> (v-100) * 4.33; 201-230 days/past-30 -> v-200;
    888 = no drinks -> 0; 777/999 -> NaN; anything else -> NaN.
    """
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out = out.mask(s.between(101, 199), (s - 100) * 4.33)
    out = out.mask(s.between(201, 299), s - 200)
    out = out.mask(s == 888, 0.0)
    return out  # 777/999 and any exotic code stay NaN


def _recode_fv(s: pd.Series) -> pd.Series:
    """Fruit/vegetable frequency -> times per month (float).

    101-199 times per DAY  -> (v-100) * 30.4
    201-299 times per WEEK -> (v-200) * 4.33
    300   = 'less than once a month' -> 0.5
    301-399 times per MONTH -> v - 300
    555   = 'never' -> 0.0
    777/999 -> NaN; anything else -> NaN.
    """
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out = out.mask(s.between(101, 199), (s - 100) * 30.4)
    out = out.mask(s.between(201, 299), (s - 200) * 4.33)
    out = out.mask(s == 300, 0.5)
    out = out.mask(s.between(301, 399), s - 300)
    out = out.mask(s == 555, 0.0)
    return out


def _recode_bmi(s: pd.Series) -> pd.Series:
    """_BMI5 has 2 implied decimals -> /100; guard implausible values."""
    out = s / 100.0
    return out.where(out.between(10.0, 99.99))


def _recode_mmyyyy(s: pd.Series) -> pd.Series:
    """FLSHTMY3: 777777 (DK) / 999999 (refused) -> NaN; MMYYYY kept (nominal)."""
    return s.where(~s.isin([777777, 999999]))


_RECODE_FNS = {
    "daycnt": _recode_daycnt,
    "drnk": _recode_drnk,
    "avedrnk": _recode_avedrnk,
    "child": _recode_child,
    "marj": _recode_marj,
    "alc": _recode_alc,
    "fv": _recode_fv,
    "bmi": _recode_bmi,
    "mmyyyy": _recode_mmyyyy,
}


def apply_recode(s: pd.Series, recode: tuple) -> pd.Series:
    """Apply one whitelist recode spec to a float Series."""
    kind = recode[0]
    if kind == "none":
        return s.astype(float)
    if kind == "sent":
        return _recode_sent(s.astype(float), recode[1]).astype(float)
    return _RECODE_FNS[kind](s.astype(float))


# [3] CLEANING  (load -> sentinel recode -> ENG transforms -> valid-target mask; no randomness anywhere in this path)
def load_raw() -> pd.DataFrame:
    """Read the immutable raw BRFSS 2021 CSV (one pass)."""
    if not DATA_RAW.exists():
        raise SystemExit(
            f"[error] raw data not found: {DATA_RAW}\n"
            f"        Expected the BRFSS 2021 LLCP file at that path "
            f"(relative to this script:\n"
            f"        deliverables/../data/raw/LLCP2021.csv).\n"
            f"        Download it from https://www.dlsi.ua.es/~juanra/UA/"
            f"datasets/LLCP2021.csv.tar.gz , extract, and place LLCP2021.csv "
            f"in data/raw/."
        )
    print(f"[clean] reading {DATA_RAW} ...")
    df = pd.read_csv(DATA_RAW, low_memory=False)
    print(f"[clean] raw shape = {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def build_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Whitelisted, sentinel-recodeed, ENG-applied feature matrix (no target).

    float64 DataFrame with the 76 whitelist columns in whitelist order and
    all raw rows (the valid-target mask is applied by build_dataset).
    """
    missing_cols = [c for c in WHITELIST if c not in df.columns]
    if missing_cols:
        raise KeyError(f"whitelist columns absent from raw data: {missing_cols}")
    parts = []
    for col, spec in WHITELIST.items():
        recoded = apply_recode(df[col], spec["recode"])
        parts.append(recoded.rename(col))
    X = pd.concat(parts, axis=1)
    # hard leakage guarantee: the blacklist can never be in the matrix
    leaked = [c for c in BLACKLIST if c in X.columns]
    if leaked:
        raise AssertionError(f"LEAKAGE: blacklisted columns in matrix: {leaked}")
    return X


def make_target(genhlth: pd.Series):
    """Map raw GENHLTH codes to the binary target.

    Returns y (0/1, int) and a boolean mask of rows with a valid target.
    Rows coded 7 (Don't know), 9 (Refused) or blank are excluded, never
    imputed.
    """
    valid = genhlth.isin(list(POSITIVE_CLASSES) + list(NEGATIVE_CLASSES))
    y = genhlth.isin(POSITIVE_CLASSES).astype(int)
    return y, valid


def build_dataset() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Full cleaning path: raw CSV -> X (float32) + y (int8) + audit info."""
    t0 = time.perf_counter()
    df = load_raw()
    assert df.shape[0] == EXPECTED_RAW_ROWS, df.shape

    X_all = build_clean(df)                       # 438,693 x 76, float64
    y_all, valid = make_target(df[TARGET_COL])
    y_valid = y_all[valid].to_numpy(dtype=np.int8)   # positional ndarray
    X = X_all.loc[valid].reset_index(drop=True).astype("float32")

    audit = sanity_checks(X_all, X, y_valid)
    audit["clean_s"] = time.perf_counter() - t0
    audit["genhlth_raw"] = df[TARGET_COL].loc[valid].to_numpy()  # 1-5 codes, EDA panel
    return X, y_valid, audit


def sanity_checks(X_all: pd.DataFrame, X: pd.DataFrame,
                  y: np.ndarray) -> dict:
    """Hard asserts on shapes, target counts and value contracts."""
    print("[clean] sanity checks ...")
    # 1. leakage blacklist absent
    assert not (set(BLACKLIST) & set(X_all.columns)), "blacklisted column present"
    # 2. shapes
    assert X_all.shape == (EXPECTED_RAW_ROWS, EXPECTED_N_FEATURES), X_all.shape
    assert X.shape == (EXPECTED_N_VALID, EXPECTED_N_FEATURES), X.shape
    # 3. target counts / class rate
    n_pos = int(y.sum())
    rate = 100.0 * n_pos / len(y)
    assert len(y) == EXPECTED_N_VALID, len(y)
    assert n_pos == EXPECTED_N_POS, n_pos
    assert abs(rate - 16.63) < 0.01, rate
    print(f"[clean]   matrix {X.shape[0]:,} x {X.shape[1]} (float32) | "
          f"{n_pos:,} positives = {rate:.2f}% class rate | "
          f"{EXPECTED_RAW_ROWS - len(y):,} rows dropped for invalid target")
    # 4. residual-sentinel / value-contract scan
    violations = []
    for col, allowed in VALID_SETS.items():
        vals = X_all[col].dropna()
        bad = vals[~vals.isin(allowed)]
        if len(bad):
            violations.append((col, "set", bad.value_counts().head(5).to_dict()))
    for col, (lo, hi) in VALID_RANGES.items():
        vals = X_all[col].dropna()
        bad = vals[(vals < lo) | (vals > hi)]
        if len(bad):
            violations.append((col, "range",
                               {"n": int(len(bad)), "min": float(bad.min()),
                                "max": float(bad.max())}))
    for col in ("PHYSHLTH", "MENTHLTH", "POORHLTH", "DRNK3GE5", "AVEDRNK3",
                "CHILDREN", "MARIJAN1"):
        assert not (X_all[col] == 88).any(), f"{col}: residual 88 sentinel"
        assert not X_all[col].isin([77, 99]).any(), f"{col}: residual 77/99"
    assert not X_all["ALCDAY5"].isin([777, 999, 888]).any(), \
        "ALCDAY5 residual sentinel"
    assert not X_all["FRUIT2"].isin([777, 999, 555, 300]).any(), \
        "FRUIT2 residual code"
    assert X_all["_STATE"].nunique() == EXPECTED_STATE_NUNIQUE
    if violations:
        for col, kind, detail in violations:
            print(f"  VIOLATION {col} ({kind}): {detail}")
        raise AssertionError(f"{len(violations)} value-contract violations")
    # 5. effective coverage gate (every kept column >= 25% non-NaN)
    cov = X_all.notna().mean()
    below = cov[cov < 0.25]
    assert below.empty, f"kept columns under 25% coverage: {below.to_dict()}"
    print(f"[clean]   value contracts hold (no residual sentinels); all "
          f"{len(WHITELIST)} columns >= 25% effective coverage "
          f"(min {cov.min() * 100:.1f}% on {cov.idxmin()})")
    return {"n_valid": len(y), "n_pos": n_pos, "rate": rate}


# [4] EXPLORATORY DATA ANALYSIS - report section 4 (target / missingness / rho)
FIGURES_DIR = ROOT / "figures"
C_NEG, C_POS = "#4C72B0", "#DD8452"     # y = 0 / y = 1 (and signed rho)
C_TOP, C_REST = "#C44E52", "#9AA5B1"    # highlight / neutral bars
GENHLTH_LBL = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}


def run_eda(X: pd.DataFrame, y: np.ndarray, genhlth_raw: np.ndarray,
            mode: str, n_raw: int) -> None:
    """Explore before modeling: the numbers cited in report section 4.

    Prints the binary target distribution, the missingness profile of the
    76-column matrix and the top univariate Spearman associations with the
    target. Full mode also regenerates the first three EDA figures into
    figures/; smoke mode prints the same tables on the subsample only.
    """
    n, n_pos = len(y), int(y.sum())
    ratio = (n - n_pos) / n_pos
    print(f"[eda] binary target: {n_pos:,} / {n:,} = {100 * y.mean():.2f}% "
          f"Fair/Poor (imbalance 1 : {ratio:.1f})")

    # missingness profile (sentinels already recoded to NaN in [3])
    miss = (X.isna().mean() * 100).sort_values(ascending=False)
    med = miss.median()
    n_complete = int((miss == 0).sum())
    n25, n50 = int((miss > 25).sum()), int((miss > 50).sum())
    print(f"[eda] missingness: median {med:.1f}% | complete {n_complete} | "
          f">25% {n25} | >50% {n50} | max {miss.index[0]} {miss.iloc[0]:.1f}%")
    print("[eda] top-10 missing: "
          + ", ".join(f"{c} {v:.1f}" for c, v in miss.head(10).items()))

    # univariate association with the target (NaN-aware Spearman)
    rho = X.corrwith(pd.Series(y, name="y"), method="spearman")
    top = rho.reindex(rho.abs().sort_values(ascending=False).index)
    print("[eda] top-15 |Spearman rho| vs target:")
    print(top.head(15).round(4).to_string())

    if plt is None or mode != "full":
        print("[eda] figures skipped (no matplotlib or smoke mode)")
        return

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10.5, "axes.grid": True, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.bbox": "tight",
    })
    FIGURES_DIR.mkdir(exist_ok=True)

    # -- figure 1: raw GENHLTH classes + the binary target --
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    ax = axes[0]
    cnts = [int((genhlth_raw == c).sum()) for c in GENHLTH_LBL]
    bars = ax.bar(range(5), cnts, color=[C_NEG] * 3 + [C_POS] * 2,
                  edgecolor="white", width=0.72)
    for b, c in zip(bars, cnts):
        ax.text(b.get_x() + b.get_width() / 2, c + 2500,
                f"{c:,}\n({c / n * 100:.1f}%)", ha="center", va="bottom",
                fontsize=10)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{c}\n{GENHLTH_LBL[c]}" for c in GENHLTH_LBL])
    ax.set_ylabel("Respondents")
    ax.set_ylim(0, max(cnts) * 1.18)
    ax.set_title("(a) Raw GENHLTH, 5 classes")
    ax.text(0.985, 0.97,
            f"valid rows n = {n:,}\n{n_raw - n:,} rows (DK/Refused/blank) excluded",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F2F2F2", ec="#CCCCCC"))
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=C_NEG),
                       Rectangle((0, 0), 1, 1, fc=C_POS)],
              labels=["y = 0  (classes 1-3)", "y = 1  (classes 4-5)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, fontsize=10)

    ax = axes[1]
    n0 = n - n_pos
    bars = ax.bar([0, 1], [n0, n_pos], color=[C_NEG, C_POS],
                  edgecolor="white", width=0.55)
    for b, c in zip(bars, [n0, n_pos]):
        ax.text(b.get_x() + b.get_width() / 2, c + 6000, f"{c:,}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, c / 2, f"{c / n * 100:.2f} %",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["y = 0\nGood or better\n(GENHLTH 1-3)",
                        "y = 1\nFair / Poor\n(GENHLTH 4-5)"])
    ax.set_ylabel("Respondents")
    ax.set_ylim(0, n0 * 1.2)
    ax.set_title(f"(b) Binary target: 1 : {ratio:.1f} class imbalance")
    fig.suptitle("Target definition and class imbalance - BRFSS 2021, "
                 "GENHLTH recoded to binary", fontsize=13.5, fontweight="bold")
    fig.savefig(FIGURES_DIR / "eda_target_imbalance.png")
    plt.close(fig)

    # -- figure 2: missingness per column, sorted (76 bars) --
    fig, ax = plt.subplots(figsize=(9.2, 13.2))
    vals = miss.to_numpy()
    ax.barh(range(len(vals)), vals,
            color=[C_TOP if i < 10 else C_REST for i in range(len(vals))],
            edgecolor="white", height=0.72)
    ax.invert_yaxis()
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(miss.index, fontsize=8)
    ax.set_xlabel("Missing values after sentinel recode (%)")
    ax.set_xlim(0, 88)
    for i in range(10):
        ax.text(vals[i] + 0.7, i + 0.02, f"{vals[i]:.1f}", va="center",
                fontsize=8, color=C_TOP)
    for v in (25, 50):
        ax.axvline(v, color="#555555", ls="--", lw=1.1)
        ax.text(v + 0.6, len(vals) - 1.2, f"{v} %", color="#555555", fontsize=9)
    ax.set_title("Missingness of the 76-column matrix v1 (valid rows, sorted)")
    ax.text(0.985, 0.03,
            f"n = {n:,} valid rows - {n50} columns > 50% - {n25} columns > 25%\n"
            f"median {med:.1f}% - {n_complete} columns complete\n"
            "top-10 highlighted (crimson): optional-module + conditional items",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FDF3F3", ec=C_TOP, lw=0.8))
    fig.savefig(FIGURES_DIR / "eda_missingness_matrix.png")
    plt.close(fig)

    # -- figure 3: top-25 |Spearman rho| with the target --
    top25 = top.head(25)[::-1]
    fig, ax = plt.subplots(figsize=(9.8, 9.4))
    vals = top25.to_numpy()
    ax.barh(range(len(vals)), vals,
            color=[C_POS if v > 0 else C_NEG for v in vals],
            edgecolor="white", height=0.72)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(top25.index, fontsize=9.5)
    for i, v in enumerate(vals):
        ax.text(v + (0.010 if v > 0 else -0.010), i, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8.5,
                color="#7a4a1e" if v > 0 else "#33507a")
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlim(-0.55, 0.58)
    ax.set_xlabel(r"Spearman $\rho$ with target (y = 1 Fair/Poor)")
    ax.set_title(r"Top-25 whitelist features by |Spearman $\rho$| with the target")
    fig.savefig(FIGURES_DIR / "eda_association_top25.png")
    plt.close(fig)
    print(f"[eda] figures written: 3 PNG in {FIGURES_DIR}")


# [5] PIPELINE - NaN-passthrough preprocessor (the booster route)
def column_groups() -> dict:
    """Column groups per encoding, in whitelist order (as in column_meta)."""
    return {
        "NOM": [c for c, spec in WHITELIST.items()
                if spec["encoding"] == "NOM"],
        "NUMERIC": [c for c, spec in WHITELIST.items()
                    if spec["encoding"] in ("ORD", "NUM", "ENG")],
    }


def build_preprocessor() -> ColumnTransformer:
    """Model-time preprocessing for the final (LightGBM) model - NO imputation.

    NOM -> OneHotEncoder(handle_unknown='ignore') directly: with
    handle_unknown='ignore', sklearn's OneHotEncoder encodes NaN as its OWN
    category (e.g. SOMALE -> categories [1.0, 2.0, 3.0, 4.0, nan]; verified
    on scikit-learn 1.9.1) and unseen NaNs at transform time are covered.
    ORD/NUM/ENG -> passthrough with NaN intact, routed natively by
    LightGBM at split time. Missingness stays signal instead of being
    blurred by most-frequent imputation - native missing handling is part
    of why boosting suits module-sparse survey data.

    Always used INSIDE the CV pipeline below, i.e. fit on training folds
    only (never on the full data).
    """
    groups = column_groups()
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


# [6] FINAL MODEL - tuned LightGBM (EXP-014a)
def build_final_model() -> LGBMClassifier:
    """Tuned LightGBM, hyperparameters verbatim from the tuning phase."""
    return LGBMClassifier(**LGBM_FIXED_FLAGS, **TUNED_LGBM_PARAMS)


def build_pipeline() -> Pipeline:
    """prep -> model, the exact pipeline scored on the canonical folds."""
    return Pipeline([("prep", build_preprocessor()),
                     ("model", build_final_model())])


# [7] EVALUATION - stratified 10-fold CV, ROC-AUC, canonical folds
def make_folds() -> StratifiedKFold:
    """The one and only CV splitter (identical to every logged run)."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                           random_state=RANDOM_STATE)


def fold_list_hash(folds) -> str:
    """sha256 over the exact fold index arrays (order-sensitive).

    Same hash function as the project harness: for each fold, the train
    then test index arrays are hashed as int64 bytes.
    """
    h = hashlib.sha256()
    for train, test in folds:
        h.update(np.asarray(train, dtype=np.int64).tobytes())
        h.update(np.asarray(test, dtype=np.int64).tobytes())
    return h.hexdigest()


def make_fold_list(X, y) -> tuple[list, str]:
    """Materialise the folds once; integrity-check they partition the rows."""
    folds = [(tr, te) for tr, te in make_folds().split(X, y)]
    all_test = np.concatenate([te for _, te in folds])
    if len(all_test) != len(X) or len(np.unique(all_test)) != len(X):
        raise AssertionError("folds do not partition the rows exactly once")
    for tr, te in folds:
        if len(np.intersect1d(tr, te)):
            raise AssertionError("train/test overlap in folds")
    return folds, fold_list_hash(folds)


def _fold_worker(fold, X, y, pipe):
    """Fit the cloned pipeline on a training fold; AUC + OOF probs on test.

    This is exactly what sklearn's cross_val_score computes per fold
    (clone -> fit -> predict_proba -> roc_auc), just keeping the
    probabilities so the OOF confusion matrix comes from the same pass.
    """
    train_idx, test_idx = fold
    p = clone(pipe)
    p.fit(X.iloc[train_idx], y[train_idx])
    proba = p.predict_proba(X.iloc[test_idx])[:, 1]
    auc = roc_auc_score(y[test_idx], proba)
    return test_idx, proba, float(auc)


def run_cv(X, y, pipe, folds) -> tuple[np.ndarray, np.ndarray, float]:
    """10-fold CV in parallel across folds (as logged: n_jobs=-1,
    pre_dispatch='n_jobs', single-threaded model inside each worker).

    Returns (fold AUCs in fold order, OOF probabilities, wall seconds).
    """
    t0 = time.perf_counter()
    results = Parallel(n_jobs=-1, pre_dispatch="n_jobs")(
        delayed(_fold_worker)(fold, X, y, pipe) for fold in folds)
    cv_s = time.perf_counter() - t0
    scores = np.array([r[2] for r in results], dtype=float)
    oof = np.full(len(X), np.nan, dtype=float)
    for test_idx, proba, _ in results:
        oof[test_idx] = proba
    assert not np.isnan(oof).any(), "OOF probabilities incomplete"
    return scores, oof, cv_s


# [8] REPORTING
def build_report(mode: str, X, y, folds, fold_hash, scores, oof, cv_s,
                 clean_s, total_s, audit) -> tuple[list, np.ndarray]:
    """Assemble the console/file report lines + the EXP-014a delta table."""
    n_pos = int(y.sum())
    rate = 100.0 * y.mean()
    mean, std = float(scores.mean()), float(scores.std(ddof=0))
    preds = (oof >= 0.5).astype(int)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    oof_auc = roc_auc_score(y, oof)
    acc = (preds == y).mean()

    lines = []
    a = lines.append
    a("=" * 72)
    a(f"BRFSS 2021 GENHLTH binary classifier - FINAL MODEL "
      f"({'FULL' if mode == 'full' else 'SMOKE'} run)")
    a(f"run at {datetime.now().isoformat(timespec='seconds')}")
    a("=" * 72)
    a("")
    a("[dataset]")
    if mode == "full":
        a(f"  rows: {len(y):,} valid-target rows of {EXPECTED_RAW_ROWS:,} raw "
          f"({EXPECTED_RAW_ROWS - len(y):,} dropped: GENHLTH in "
          f"{{7, 9, blank}})")
    else:
        a(f"  rows: {len(y):,} stratified subsample "
          f"(SMOKE mode; full data = {EXPECTED_N_VALID:,} rows)")
    a(f"  features: {X.shape[1]} (ADR-005 whitelist; "
      f"{len(column_groups()['NOM'])} nominal one-hot, "
      f"{len(column_groups()['NUMERIC'])} ordinal/numeric/engineered)")
    a(f"  class rate: {n_pos:,} positives = {rate:.2f}% "
      f"(imbalance 1 : {(len(y) - n_pos) / n_pos:.2f})")
    a("")
    a("[model]")
    a("  LGBMClassifier (tuned, EXP-014a) inside NaN-passthrough pipeline:")
    for k in sorted({**LGBM_FIXED_FLAGS, **TUNED_LGBM_PARAMS}):
        a(f"    {k} = {({**LGBM_FIXED_FLAGS, **TUNED_LGBM_PARAMS})[k]}")
    a("  preprocessing: NOM -> OneHotEncoder(handle_unknown='ignore'); "
      "ORD/NUM/ENG -> passthrough (NaN kept; native LightGBM routing)")
    a("")
    a("[folds]")
    a(f"  StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, "
      f"random_state={RANDOM_STATE})")
    a(f"  folds sha256 = {fold_hash}")
    if mode == "full":
        a(f"  canonical-prefix check: {fold_hash[:16]} == "
          f"{CANONICAL_FOLDS_SHA16} -> "
          f"{'MATCH (identical to all logged experiments)' if fold_hash.startswith(CANONICAL_FOLDS_SHA16) else 'MISMATCH'}")
    else:
        a("  (smoke-mode folds are over the 50k subsample - hash differs "
          "from the canonical full-data folds by construction)")
    a("")
    a("[fold table]")
    header = (f"  {'fold':>4s} {'n_test':>8s} {'pos_test':>8s} "
              f"{'ROC-AUC':>10s}")
    if mode == "full":
        header += f" {'EXP-014a':>10s} {'delta':>11s}"
    a(header)
    deltas = np.full(N_SPLITS, np.nan)
    for i, ((_, te), s) in enumerate(zip(folds, scores), start=1):
        row = (f"  {i:4d} {len(te):8,d} {int(y[te].sum()):8,d} {s:10.6f}")
        if mode == "full":
            d = s - EXP014A_FOLDS[i - 1]
            deltas[i - 1] = d
            row += f" {EXP014A_FOLDS[i - 1]:10.6f} {d:+11.2e}"
        a(row)
    a(f"  mean CV ROC-AUC = {mean:.6f} +/- {std:.6f} (std over "
      f"{N_SPLITS} folds, ddof=0)")
    if mode == "full":
        a(f"  logged EXP-014a mean = {EXP014A_MEAN:.6f} | "
          f"mean delta = {mean - EXP014A_MEAN:+.2e}")
        max_abs = float(np.abs(deltas).max())
        verdict = ("PASS" if max_abs <= MATCH_TOL else "EXCEEDED")
        a(f"  max |per-fold delta| vs logged EXP-014a = {max_abs:.2e} "
          f"(tolerance {MATCH_TOL:.0e}) -> {verdict}")
    a("")
    a("[out-of-fold diagnostics]")
    a("  confusion matrix at threshold 0.5 on OOF predictions "
      f"({len(oof):,} rows):")
    a("                  predicted 0     predicted 1")
    a(f"    actual 0   {tn:12,d}   {fp:12,d}")
    a(f"    actual 1   {fn:12,d}   {tp:12,d}")
    a(f"  OOF accuracy @0.5 = {acc:.4f} | OOF pooled ROC-AUC = "
      f"{oof_auc:.6f}")
    a(f"  OOF recall (TPR) = {tp / (tp + fn):.4f} | OOF precision = "
      f"{tp / (tp + fp):.4f}")
    a("")
    a("[runtime]")
    a(f"  load + clean: {clean_s:.1f} s | 10-fold CV: {cv_s:.1f} s | "
      f"total: {total_s:.1f} s ({total_s / 60:.1f} min)")
    a("")
    a("[environment]")
    import sys
    a(f"  Python {sys.version.split()[0]} | scikit-learn "
      f"{SKLEARN_VERSION} | lightgbm {lightgbm.__version__} | "
      f"pandas {pd.__version__} | numpy {np.__version__}")
    a("  joblib Parallel across folds (n_jobs=-1, pre_dispatch='n_jobs'); "
      "LightGBM n_jobs=1 per fold")
    a("")
    a("[traceability]")
    a("  Final model = EXP-014a in results/experiments_log.csv (tuned via")
    a("  RandomizedSearchCV; params verbatim from results/p6/best_params.json).")
    a("  Unbiased nested-CV estimate of the same tuned pipeline: EXP-015")
    a("  (mean 0.904303). Baselines on the same folds: Dummy 0.500000")
    a("  (EXP-001), LogisticRegression C=0.01 0.892588 (EXP-002a), best")
    a("  single DecisionTree 0.880206 (EXP-003i), untuned LGBM 0.903355")
    a("  (EXP-006).")
    a("=" * 72)
    return lines, deltas


def write_outputs(lines, folds, y, scores, deltas) -> None:
    """Save results/final_run_summary.txt + results/final_run_fold_scores.csv
    (full mode only - smoke runs never clobber the full-run artifacts)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {SUMMARY_TXT}")
    rows = []
    for i, ((_, te), s) in enumerate(zip(folds, scores), start=1):
        rows.append({
            "fold": i,
            "n_test": len(te),
            "n_pos_test": int(y[te].sum()),
            "roc_auc": f"{s:.6f}",
            "roc_auc_mean": f"{scores.mean():.6f}",
            "roc_auc_std": f"{scores.std(ddof=0):.6f}",
            "exp014a_logged": f"{EXP014A_FOLDS[i - 1]:.6f}",
            "delta_vs_exp014a": f"{deltas[i - 1]:+.2e}",
        })
    pd.DataFrame(rows).to_csv(FOLD_SCORES_CSV, index=False, encoding="utf-8")
    print(f"[report] wrote {FOLD_SCORES_CSV}")


# [9] MAIN
def main() -> None:
    ap = argparse.ArgumentParser(
        description="BRFSS 2021 GENHLTH binary classifier - final model "
                    "(tuned LightGBM, EXP-014a) on the canonical folds.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--full", action="store_true", default=True,
                   help=f"full run: all {EXPECTED_N_VALID:,} valid rows "
                        "(default)")
    g.add_argument("--smoke", action="store_true",
                   help=f"fast verification: stratified {SMOKE_N:,}-row "
                        f"subsample (random_state={RANDOM_STATE})")
    args = ap.parse_args()
    mode = "smoke" if args.smoke else "full"

    t_start = time.perf_counter()
    print(f"[main] mode = {mode}")

    # ---- cleaning (from the immutable raw CSV) ----
    X, y, audit = build_dataset()
    raw_codes = audit.pop("genhlth_raw")   # GENHLTH 1-5 codes for the EDA panel

    # ---- smoke subsample (verification only; full mode uses everything) --
    if mode == "smoke":
        idx = np.arange(len(X))
        sub_idx, _ = train_test_split(idx, train_size=SMOKE_N, stratify=y,
                                      random_state=RANDOM_STATE)
        X = X.iloc[sub_idx].reset_index(drop=True)
        y = y[sub_idx]
        raw_codes = raw_codes[sub_idx]
        print(f"[main] smoke subsample: {len(y):,} rows "
              f"({int(y.sum()):,} positives = {100.0 * y.mean():.2f}%)")

    # ---- exploratory profile of exactly the rows being modeled ----
    run_eda(X, y, raw_codes, mode, n_raw=EXPECTED_RAW_ROWS)

    # ---- canonical folds ----
    folds, fold_hash = make_fold_list(X, y)
    # determinism proof: rebuilding the splitter reproduces the same folds
    _, fold_hash_again = make_fold_list(X, y)
    assert fold_hash_again == fold_hash, "make_folds() not deterministic"
    print(f"[main] folds sha256 = {fold_hash}")
    if mode == "full":
        assert fold_hash.startswith(CANONICAL_FOLDS_SHA16), (
            f"canonical fold hash mismatch: {fold_hash[:16]} != "
            f"{CANONICAL_FOLDS_SHA16} - raw data or sklearn version differs "
            "from the logged experiments")
        print(f"[main] canonical folds confirmed "
              f"({fold_hash[:16]} == {CANONICAL_FOLDS_SHA16}); identical to "
              "every logged experiment (EXP-001..EXP-015)")

    # ---- evaluate the final model ----
    pipe = build_pipeline()
    print(f"[main] running {N_SPLITS}-fold CV "
          f"(LightGBM n_estimators={TUNED_LGBM_PARAMS['n_estimators']}, "
          f"num_leaves={TUNED_LGBM_PARAMS['num_leaves']}) ...")
    scores, oof, cv_s = run_cv(X, y, pipe, folds)

    # ---- report (+ artifacts in full mode) ----
    total_s = time.perf_counter() - t_start
    lines, deltas = build_report(mode, X, y, folds, fold_hash, scores, oof,
                                 cv_s, audit["clean_s"], total_s, audit)
    print("\n".join(lines))
    if mode == "full":
        write_outputs(lines, folds, y, scores, deltas)
        print("[main] full run complete - artifacts in results/")
    else:
        print("[main] smoke run complete (no artifacts written; use --full "
              "for the recorded run)")


if __name__ == "__main__":
    main()
