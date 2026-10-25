"""Shared configuration for the BRFSS GENHLTH project.

All experiments import from here. These values are FIXED project constants —
changing any of them requires an ADR (see vault/30 Decisions/).
"""
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

# ---- Project constants (immutable without an ADR) ----
RANDOM_STATE = 42
N_SPLITS = 10
POSITIVE_CLASSES = (4, 5)          # y = 1 if GENHLTH in {4,5} (Fair/Poor)
NEGATIVE_CLASSES = (1, 2, 3)       # y = 0 if GENHLTH in {1,2,3}
TARGET_COL = "GENHLTH"
DROP_TARGET_CODES = (7, 9)         # Don't know / Refused -> row dropped, not imputed

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "LLCP2021.csv"
DATA_INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
EXPERIMENTS_LOG = RESULTS / "experiments_log.csv"

for _d in (DATA_INTERIM, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


def make_folds() -> StratifiedKFold:
    """The one and only CV splitter used anywhere in this project.

    Every model, feature-selection run, resampling pipeline and statistical
    test must use folds from this configuration so scores are comparable.
    """
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def make_target(genhlth):
    """Map raw GENHLTH codes to the binary target.

    Returns y (0/1) and a boolean mask of rows with a valid target.
    Rows coded 7 (Don't know), 9 (Refused) or blank are excluded.
    """
    valid = genhlth.isin(list(POSITIVE_CLASSES) + list(NEGATIVE_CLASSES))
    y = genhlth.isin(POSITIVE_CLASSES).astype(int)
    return y, valid
