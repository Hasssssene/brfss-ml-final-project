"""P2 - cleaned feature matrix v1 + frozen column whitelist (ADR-005).

Builds the whitelist-only feature matrix from the immutable raw BRFSS 2021 CSV
(``data/raw/LLCP2021.csv``, 438,693 rows x 303 columns) by:

1. enforcing the ADR-005 column whitelist (76 columns; every one of the 303
   census columns has an explicit verdict + reason, see ``results/profiling/
   triage_decisions.csv``),
2. recoding BRFSS missing-value sentinels to NaN per-column (exact codebook
   sentinel sets - never ``7``/``9`` where those are legitimate values, e.g.
   EMPLOY1 7 = "Retired", INCOME3 7/9 = income brackets, _RACE 7 =
   "Multiracial", PRIMINSR 7/9 = TRICARE / state plan),
3. applying the ENG recodes (day-count 88->0, ALCDAY5 / fruit-vegetable
   frequency decode, BMI implied decimals).

Deliverables written by ``__main__``:

- ``data/interim/clean_v1.parquet``     feature matrix X (whitelisted cols,
  sentinels -> NaN, ENG applied). All 438,693 rows, float64; NO target column.
- ``data/interim/target_v1.parquet``    row-aligned y (int8) + valid mask
  (bool) + raw GENHLTH for audit; built via ``src.config.make_target``.
- ``data/interim/column_meta.json``     per kept column: encoding, recode,
  verdict, rule, reason, effective coverage, missingness.
- ``results/profiling/triage_decisions.csv``  all 303 columns: verdict,
  rule, reason, coverage.

Verdict precedence (documented implementation choice, rules as in ADR-005):
rule 1 DROP-LEAK  >  rule 3 DROP-CONSTANT  >  rule 2 DROP-ID / DROP-MODE  >
rule 6 DROP-DUP  >  rule 4 DROP-SPARSE (>90% blank+sentinel)  >
rule 5 DROP-MODULE (<25% effective coverage)  >  KEEP.
The 11 census-constant columns + IYEAR carry the DROP-CONSTANT verdict per
rule 3 (8 of them also sit in the rule-2 flow block - noted in the reason).

Rule-7 judgment calls (recorded in triage_decisions.csv, no silent decisions):
- QSTVER dropped (instrument version admin); QSTLANG kept (interview language
  is a respondent attribute).
- _METSTAT dropped as the urbanicity sibling of kept _URBSTAT (one per pair).
- MSCODE dropped (MSA geo-stratum, weighting admin, redundant with _STATE).
- NUMHHOL3 / CPDEMO1B dropped as telephone-admin (rule 2 "etc.": dual-frame
  weighting inputs) even though NUMHHOL3 alone passes the 25% coverage rule.
- HTIN4 / HTM4 / WTKG3 dropped (linear metric transforms of dropped
  HEIGHT3 / WEIGHT2; BMI family per rule 6).
- AVEDRNK3 / MAXDRNKS kept (raw alcohol items not enumerated in rule 6;
  "prefer the raw/simplest representation").
- FLSHTMY3 kept as nominal (raw immunization item surviving rule 5; MMYYYY
  codes incl. 37 malformed-but-real ones retained as categories).
- CHOLCHK3 kept as NOMINAL, not ordinal: codebook order is non-monotone
  (1 = "Never" vs 2-8 = decreasing recency).
- CAREGIV1 value 8 ("caregiving recipient died", 328 rows) is a valid
  codebook response - kept as its own nominal category.

Sentinel recodes verified against docs/codebook.pdf (LLCP 2021 codebook) and
``results/profiling/column_census.csv`` top values. The census's generic
7/77/9/99 exact-match scan OVERCOUNTS sentinels for columns where those are
valid codes (EMPLOY1, INCOME3, JOINPAI2, PRIMINSR, IMFVPLA2, CHILDREN,
MAXDRNKS, AVEDRNK3, _RACE, _STATE, _AGE80, ...); this module uses the
codebook-exact sets instead. Effective coverage below is therefore computed
from the recoded matrix, not from census percentages.

No randomness is used in the cleaning path. The optional smoke test at the
end uses RANDOM_STATE from src/config.py (subsample + LogisticRegression).
It is NOT an experiment and is NOT logged to results/experiments_log.csv.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from src.config import (  # noqa: E402
    DATA_INTERIM,
    DATA_RAW,
    RESULTS,
    RANDOM_STATE,
    TARGET_COL,
    make_target,
)

PROFILING_DIR = RESULTS / "profiling"
CENSUS_CSV = PROFILING_DIR / "column_census.csv"
CLEAN_PARQUET = DATA_INTERIM / "clean_v1.parquet"
TARGET_PARQUET = DATA_INTERIM / "target_v1.parquet"
META_JSON = DATA_INTERIM / "column_meta.json"
TRIAGE_CSV = PROFILING_DIR / "triage_decisions.csv"

# Rule 1 - DROP-LEAK: the target source and its calculated derivatives.
# Hard blacklist; asserted absent from the matrix no matter what else changes.
BLACKLIST: tuple[str, ...] = ("GENHLTH", "_RFHLTH", "_PHYS14D", "_MENT14D")

# Rule 3 - DROP-CONSTANT: the 11 census-constant columns + IYEAR (2 values).
# (8 of the 11 are also rule-2 landline/cell flow columns - reason notes it.)
CONSTANT_COLS: tuple[str, ...] = (
    "COLGHOUS", "STATERE1", "CELPHON1", "SAFETIME", "CTELNUM1", "CELLFON5",
    "CADULT1", "CCLGHOUS", "TOLDCFS", "HAVECFS", "WORKCFS", "IYEAR",
)

# Rule 2 - DROP-ID: identifiers, dates, dispatch, weights/strata, instrument
# admin. QSTVER + MSCODE are rule-7 judgment calls (see module docstring).
ID_COLS: tuple[str, ...] = (
    "FMONTH", "IDATE", "IMONTH", "IDAY", "DISPCODE", "SEQNO", "_PSU",
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_LLCPWT2",
    "_LLCPWT", "_DUALUSE", "_DUALCOR", "QSTVER", "MSCODE",
)

# Rule 2 - DROP-MODE: landline/cell intro & flow block (explicit ADR-005 list
# + "etc.": NUMHHOL3, CPDEMO1B telephone/dual-frame weighting inputs).
MODE_COLS: tuple[str, ...] = (
    "CTELENM1", "PVTRESD1", "LADULT1", "COLGSEX", "NUMADULT", "LANDSEX",
    "NUMMEN", "NUMWOMEN", "RESPSLCT", "CELLSEX", "PVTRESD3", "CSTATE1",
    "LANDLINE", "HHADULT", "NUMHHOL3", "CPDEMO1B",
)

# Rule 6 - DROP-DUP: one side of every duplicate pair / computed family kept
# per ADR-005 (BMI, smoking, age, race, edu, income, insurance, activity,
# arthritis, alcohol, e-cig, immunization flags, FV indices, sex, BP/chol/
# asthma/MI flags, urbanicity sibling).
DUP_COLS: tuple[str, ...] = (
    # BMI family -> keep _BMI5
    "HEIGHT3", "WEIGHT2", "HTIN4", "HTM4", "WTKG3", "_BMI5CAT", "_RFBMI5",
    # smoking -> keep _SMOKER3
    "SMOKE100", "SMOKDAY2", "_RFSMOK3",
    # age -> keep _AGE80
    "_AGEG5YR", "_AGE65YR", "_AGE_G",
    # race/ethnicity -> keep _RACE
    "_PRACE1", "_MRACE1", "_HISPANC", "_RACEG21", "_RACEGR3", "_RACEPRV",
    "_IMPRACE",
    # education / income / children -> keep EDUCA, INCOME3, CHILDREN
    "_EDUCAG", "_INCOMG1", "_CHLDCNT",
    # insurance -> keep PRIMINSR
    "_HLTHPLN", "_HCVU652",
    # physical activity -> keep EXERANY2 (EXERANY2 == _TOTINDA, EXP-000)
    "_TOTINDA",
    # arthritis -> keep HAVARTH5 (+ raw follow-ups)
    "_DRDXAR3", "_LMTACT3", "_LMTWRK3",
    # alcohol -> keep decoded ALCDAY5 / AVEDRNK3 / DRNK3GE5 / MAXDRNKS
    "DRNKANY5", "DROCDY3_", "_RFBING5", "_DRNKWK1", "_RFDRHV7",
    # e-cigarette -> keep ECIGNOW1
    "_CURECI1",
    # immunization / HIV computed flags -> keep raw items
    "_FLSHOT7", "_PNEUMO3", "_AIDTST4",
    # fruits & vegetables computed indices -> keep raw frequency items
    "FTJUDA2_", "FRUTDA2_", "GRENDA1_", "FRNCHDA_", "POTADA1_", "VEGEDA2_",
    "_MISFRT1", "_MISVEG1", "_FRTRES1", "_VEGRES1", "_FRUTSU1", "_VEGESU1",
    "_FRTLT1A", "_VEGLT1A", "_FRT16A", "_VEG23A", "_FRUITE1", "_VEGETE1",
    # sex -> keep SEXVAR
    "_SEX",
    # BP / cholesterol / asthma / MI computed flags -> keep raw items
    "_RFHYPE6", "_CHOLCH3", "_RFCHOL3", "_MICHD", "_LTASTH1", "_CASTHM1",
    "_ASTHMS1",
    # urbanicity -> keep _URBSTAT (rule 7: one of the _METSTAT/_URBSTAT pair)
    "_METSTAT",
)

# Rule 4 - DROP-SPARSE: >90% blank + sentinel (census evidence; micro-modules:
# diabetes management, hepatitis, HPV, tetanus, shingles, lung-cancer &
# cancer screening, cancer survivorship, BP-advice, caregiver detail,
# firearms, marijuana follow-ups, phone count).
SPARSE_COLS: tuple[str, ...] = (
    "NUMPHON3",
    "INSULIN1", "BLDSUGAR", "FEETCHK3", "DOCTDIAB", "CHKHEMO3", "FEETCHK",
    "EYEEXAM1", "DIABEYE", "DIABEDU",
    "TOLDHEPC", "TRETHEPC", "PRIRHEPC", "HAVEHEPC", "HAVEHEPB", "MEDSHEPB",
    "HPVADVC4", "HPVADSHT", "TETANUS1", "SHINGLE2",
    "LCSFIRST", "LCSLAST", "LCSNUMCG", "LCSCTSCN",
    "HADMAM", "HOWLONG", "CERVSCRN", "CRVCLCNC", "CRVCLPAP", "CRVCLHPV",
    "HADHYST2", "PSATEST1", "PSATIME1", "PCPSARS2", "PCSTALK",
    "HADSIGM4", "COLNSIGM", "COLNTES1", "SIGMTES1", "LASTSIG4", "COLNCNCR",
    "VIRCOLO1", "VCLNTES1", "SMALSTOL", "STOLTEST", "STOOLDN1", "BLDSTFIT",
    "SDNATES1",
    "CNCRDIFF", "CNCRAGE", "CNCRTYP1",
    "CSRVTRT3", "CSRVDOC1", "CSRVSUM", "CSRVRTRN", "CSRVINST", "CSRVINSR",
    "CSRVDEIN", "CSRVCLIN", "CSRVPAIN", "CSRVCTL2",
    "WHEREBP", "SHAREBP", "WTCHSALT", "DRADVISE", "CIMEMLOS",
    "CDHOUSE", "CDASSIST", "CDHELP", "CDSOCIAL", "CDDISCUS",
    "CRGVREL4", "CRGVLNG1", "CRGVHRS1", "CRGVPRB3", "CRGVALZD", "CRGVPER1",
    "CRGVHOU1",
    "USEMRJN3", "RSNMRJN2", "LASTSMK2", "STOPSMK2",
    "FIREARM5", "GUNLOAD", "LOADULK2", "CASTHNO2",
)

# Rule 5 - DROP-MODULE: 10-25% effective coverage after sentinel recode
# (optional modules asked of a minority; ACEs get no special treatment).
LOWCOV_COLS: tuple[str, ...] = (
    "ASTHNOW", "DIABAGE3", "PREGNANT", "HIVTSTD3", "PDIABTST",
    "ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC", "ACEPUNCH",
    "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX", "ACEADSAF",
    "ACEADNED",
    "RCSGENDR", "RCSRLTN2", "CASTHDX2", "BIRTHSEX",
    "_CHISPNC", "_CRACE1", "_CPRACE1", "CAGEG",
)

# ---- the whitelist: col -> (encoding, recode) -------------------------------
# encoding: ORD (ordinal codes, natural order) | NOM (one-hot at train time)
#           NUM (plain numeric) | ENG (engineered recode applied here)
# recode:   ("none",) | ("sent", codes) | named engineered recodes
# Sentinel sets are codebook-exact (docs/codebook.pdf, verified 2026-10-25).
_WHITELIST_ROWS: tuple[tuple[str, str, tuple], ...] = (
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
    ("ALCDAY5", "ENG", ("alc",)),                  # decode to drinks/month
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
    # -- module survivors (>= 25% effective coverage, rule 5) --
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
    ("QSTLANG", "NOM", ("none",)),                 # rule 7 judgment: keep
    ("_URBSTAT", "NOM", ("none",)),
    # -- calculated keepers --
    ("_BMI5", "ENG", ("bmi",)),                    # /100, 2 implied decimals
    ("_RACE", "NOM", ("sent", (9,))),              # 1-8; 7 = Multiracial valid
    ("_AGE80", "NUM", ("none",)),                  # 18-80, imputed, no sentinels
)
WHITELIST: dict[str, dict] = {
    col: {"encoding": enc, "recode": recode}
    for col, enc, recode in _WHITELIST_ROWS
}

# Post-recode validity contracts (asserted in the runner): every non-NaN value
# must be in the allowed set / range. This doubles as the residual-sentinel
# scan - a missed sentinel would violate the allowed set.
VALID_SETS: dict[str, set] = {
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
VALID_RANGES: dict[str, tuple[float, float]] = {
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
EXPECTED_STATE_NUNIQUE = 53  # 50 states + DC + PR + GU (FIPS 9 = CT is valid)


# ---- recode implementations --------------------------------------------------
def _recode_sent(s: pd.Series, codes: tuple[int, ...]) -> pd.Series:
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
    """ALCDAY5 -> drinks per month (float).

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


_RECODE_FNS: dict[str, Callable[[pd.Series], pd.Series]] = {
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


# ---- loaders / builders --------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Read the immutable raw CSV (one pass)."""
    print(f"[p2] reading {DATA_RAW} ...")
    df = pd.read_csv(DATA_RAW, low_memory=False)
    print(f"[p2] raw shape = {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def build_clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Whitelisted, sentinel-recodeed, ENG-applied feature matrix (no target).

    Returns a float64 DataFrame with the 76 whitelist columns in whitelist
    order and all 438,693 rows (the valid-target mask is applied downstream).
    """
    if df is None:
        df = load_raw()
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


def build_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Row-aligned y + valid mask (+ raw GENHLTH for audit) via config.make_target."""
    y, valid = make_target(df[TARGET_COL])
    return pd.DataFrame(
        {"y": y.astype(np.int8), "valid": valid.astype(bool),
         "genhlth_raw": df[TARGET_COL].astype(float)}
    )


# ---- verdict table -------------------------------------------------------------
def build_verdicts() -> dict[str, dict]:
    """Every census column -> verdict/rule/reason/encoding/recode (ADR-005)."""
    v: dict[str, dict] = {}

    def add(cols, verdict, rule, reason):
        for c in cols:
            if c in v:
                raise ValueError(f"duplicate verdict for {c}")
            v[c] = {"verdict": verdict, "rule": rule, "reason": reason,
                    "encoding": "", "recode": ""}

    add(BLACKLIST, "DROP-LEAK", "R1",
        "rule 1: target source / calculated from GENHLTH (leakage blacklist; "
        "_RFHLTH = 100% agreement with y, EXP-000)")
    add(CONSTANT_COLS, "DROP-CONSTANT", "R3",
        "rule 3: census-constant (<=1 distinct value); 8 of these are also "
        "rule-2 landline/cell flow columns" )
    add(ID_COLS, "DROP-ID", "R2",
        "rule 2: identifier / date / dispatch / weight / stratum / instrument "
        "admin (no health content)")
    add(MODE_COLS, "DROP-MODE", "R2",
        "rule 2: landline/cell intro-flow block (interview-mode artifact)")
    add(DUP_COLS, "DROP-DUP", "R6",
        "rule 6: duplicate of a kept representation (computed flag / binned / "
        "linear transform of a kept raw column)")
    add(SPARSE_COLS, "DROP-SPARSE", "R4",
        "rule 4: >90% blank+sentinel (micro-coverage module artifact)")
    add(LOWCOV_COLS, "DROP-MODULE", "R5",
        "rule 5: <25% effective coverage after sentinel recode")

    # explicit reason overrides - rule 7 requires judgment calls and key
    # duplicate-pair resolutions to be recorded, not folded into boilerplate
    RO: dict[str, str] = {
        "GENHLTH": "rule 1: THE target source (y = GENHLTH in {4,5}); rho=+0.672 vs y (EXP-000); never a feature",
        "_RFHLTH": "rule 1: calculated directly from GENHLTH (fair/poor flag) - empirically 100.0% agreement with y (EXP-000); it IS the target",
        "_PHYS14D": "rule 1: binned PHYSHLTH (GENHLTH-family derived; leakage blacklist)",
        "_MENT14D": "rule 1: binned MENTHLTH (GENHLTH-family derived; leakage blacklist)",
        "IYEAR": "rule 3: 2 distinct values (2021/2022 fielding spillover), admin",
        "QSTVER": "rule 7 judgment: questionnaire-version instrument admin (state module set) - no respondent content",
        "MSCODE": "rule 7 judgment: MSA geo-stratum admin (weighting geography); redundant with kept _STATE + _URBSTAT",
        "NUMHHOL3": "rule 7 judgment under rule 2 'etc.': residential-telephone count = dual-frame weighting admin (passes rule 5 at 26.6% but is telephone logistics, not health)",
        "CPDEMO1B": "rule 7 judgment under rule 2 'etc.': cell-phone usage share, the _DUALUSE weighting input (interview-mode admin)",
        "_METSTAT": "rule 7 judgment: urbanicity sibling of kept _URBSTAT - keep one representation per pair",
        "_TOTINDA": "rule 6: PROVEN identical to kept EXERANY2 (EXP-000: identical n = 436,659, identical rho = 0.2465)",
        "_DRDXAR3": "rule 6: PROVEN identical to kept HAVARTH5 (EXP-000: identical n = 434,721, identical rho = -0.2172, identical 0.66% missing)",
        "_SEX": "rule 6: identical information to kept SEXVAR (watchlist Tier 2)",
        "_CHLDCNT": "rule 6 family (rule 7): computed binning of kept CHILDREN",
        "_HCVU652": "rule 6 family (rule 7): computed no-coverage flag from kept PRIMINSR (+ age)",
        "_RFHYPE6": "rule 6: computed BP flag from kept BPHIGH6",
        "_CHOLCH3": "rule 6: computed flag from kept CHOLCHK3",
        "_RFCHOL3": "rule 6: computed flag from kept TOLDHI3",
        "_MICHD": "rule 6: computed MI/CHD composite of kept CVDINFR4 + CVDCRHD4",
        "_LTASTH1": "rule 6: computed flag from kept ASTHMA3",
        "_CASTHM1": "rule 6: computed flag from ASTHMA3 + ASTHNOW",
        "_ASTHMS1": "rule 6: computed 3-level asthma status from ASTHMA3/ASTHNOW",
        "_LMTACT3": "rule 6: computed flag from kept LMTJOIN3",
        "_LMTWRK3": "rule 6: computed flag from kept ARTHDIS2",
        "DRNKANY5": "rule 6: computed any-alcohol flag from kept ALCDAY5 (888<->2 exact match)",
        "DROCDY3_": "rule 6: computed drink-occasions-per-day from kept ALCDAY5",
        "_RFBING5": "rule 6: computed binge flag from kept DRNK3GE5",
        "_DRNKWK1": "rule 6: computed weekly drinks (ADR-005 explicit drop)",
        "_RFDRHV7": "rule 6: computed heavy-drinking flag (ADR-005 explicit drop)",
        "_CURECI1": "rule 6: computed e-cig flag from kept ECIGNOW1",
        "_FLSHOT7": "rule 6: computed flag from kept FLUSHOT7",
        "_PNEUMO3": "rule 6: computed flag from kept PNEUVAC4",
        "_AIDTST4": "rule 6: computed flag from kept HIVTST7",
        "HTIN4": "rule 6 family (rule 7): metric height = linear transform of dropped HEIGHT3; BMI is the kept representation",
        "HTM4": "rule 6 family (rule 7): metric height = linear transform of dropped HEIGHT3",
        "WTKG3": "rule 6 family (rule 7): metric weight = linear transform of dropped WEIGHT2",
        "_INCOMG1": "rule 6: binned income - keep INCOME3 (11 levels, ordinal)",
        "_EDUCAG": "rule 6: binned education - keep EDUCA (6 levels, ordinal)",
        "_HLTHPLN": "rule 6: any-coverage flag - keep PRIMINSR (11-category detail)",
        "SMOKE100": "rule 6: raw smoking input - keep _SMOKER3 (ordinal 1-4 composite)",
        "SMOKDAY2": "rule 6: raw smoking input - keep _SMOKER3",
        "HEIGHT3": "rule 6: raw height - keep _BMI5 (div 100, numeric)",
        "WEIGHT2": "rule 6: raw weight - keep _BMI5",
        "_BMI5CAT": "rule 6: binned BMI - keep _BMI5 (continuous)",
        "_RFBMI5": "rule 6: BMI flag - keep _BMI5",
        "_AGEG5YR": "rule 6: age bins - keep _AGE80 (numeric, imputed, no sentinels)",
        "_AGE65YR": "rule 6: age bins - keep _AGE80",
        "_AGE_G": "rule 6: age bins - keep _AGE80",
        "_PRACE1": "rule 6: raw race detail - keep _RACE (single nominal grouping)",
        "_MRACE1": "rule 6: raw race detail - keep _RACE",
        "_HISPANC": "rule 6: Hispanic origin - absorbed by kept _RACE (code 8)",
        "_RACEG21": "rule 6: race grouping - keep _RACE",
        "_RACEGR3": "rule 6: race grouping - keep _RACE",
        "_RACEPRV": "rule 6: race grouping (provider) - keep _RACE",
        "_IMPRACE": "rule 6: imputed race - keep _RACE",
    }
    for col, reason in RO.items():
        v[col]["reason"] = reason
    # rule-5 specials: evidence nuance worth recording verbatim
    # HIVTSTD3: the census scan missed its 777777 (DK) / 999999
    # (refused) sentinels, so census coverage (29.2%) overstates the truth;
    # effective coverage after recode is ~20.3% < 25%.
    v["HIVTSTD3"]["reason"] = (
        "rule 5: <25% effective coverage - census 29.2% overstates it because "
        "the EXP-000 scan missed the 777777 (DK) / 999999 (refused) sentinels "
        "(38,035 + 779 rows); true coverage ~20.3%"
    )
    v["PDIABTST"]["reason"] = (
        "rule 5: 24.86% effective coverage - just under the 25% gate "
        "(borderline call resolved by the rule, not discretion)"
    )
    # keep-side reason overrides for the judgment calls worth recording
    KEEP_RO: dict[str, str] = {
        "PHYSHLTH": "ADR-004 keep (distinct construct; 88->0 mandatory); coverage 96.2% after recode",
        "MENTHLTH": "ADR-004 keep (distinct construct; 88->0 mandatory)",
        "POORHLTH": "ADR-004 keep - sentinel->NaN + imputation; final inclusion decided empirically in P5 selection",
        "CHECKUP1": "rule 5/6 keep as ORDINAL: 1<2<3<4 time buckets, 8 = 'never' kept as its own high value (ADR-005 spec), 7/9 -> NaN",
        "CHOLCHK3": "rule 7 judgment: kept as NOMINAL, not ordinal - codebook order is non-monotone (1 = 'Never' vs 2-8 = decreasing recency); one-hot preserves everything",
        "PRIMINSR": "keep per rule 6; sentinels 77/99 only - 7 (TRICARE/VA) and 9 (state plan) are VALID plans; 88 'no coverage' kept as category",
        "EMPLOY1": "keep; sentinel 9 only - code 7 = 'Retired' is VALID (census miscounted 131,546 retirees as sentinels)",
        "INCOME3": "keep as ORDINAL 1-11; sentinels 77/99 - 7 and 9 are valid income brackets (census miscount)",
        "CHILDREN": "ENG: 88 = none -> 0, 99 = refused -> NaN (codebook has NO 77 code); 7/9 children are valid",
        "JOINPAI2": "keep as NUM (0-10 scale); sentinels 77/99 only - 7 and 9 are valid pain ratings (census miscount); 31.9% effective coverage",
        "_SMOKER3": "rule 6 winner: ordinal 1-4 smoker status; 9 = DK/refused/missing -> NaN",
        "ALCDAY5": "ENG decode: 101-107 days/week x4.33, 201-230 days/month, 888 (no drinks) -> 0, 777/999 -> NaN",
        "AVEDRNK3": "rule 7 keep (raw alcohol item); ENG: 88 'none' -> 0, 77/99 -> NaN; 7/9 drinks valid",
        "DRNK3GE5": "ENG: 88 'none' -> 0, 77/99 -> NaN (ADR-005 spec)",
        "MAXDRNKS": "rule 7 keep (raw alcohol item, no 88 code); 77/99 -> NaN",
        "FLSHTMY3": "rule 7 keep as NOMINAL: raw MMYYYY immunization item surviving rule 5 (44.3% effective coverage); 777777/999999 -> NaN; 37 malformed-but-real codes (772020, 2020...) retained as categories",
        "IMFVPLA2": "keep; sentinels 77/99 only - 7 (ER) and 9 (other place) are VALID among codes 1-12",
        "FRUIT2": "ENG decode to times/month: 1xx/day x30.4, 2xx/week x4.33, 300 '<1/mo' -> 0.5, 3xx/mo, 555 'never' -> 0, 777/999 -> NaN",
        "PREDIAB1": "rule 5 keep: 25.98% coverage - just above the 25% gate",
        "SOMALE": "rule 5 keep: 25.23% coverage - the lowest-coverage kept column",
        "CAREGIV1": "keep; value 8 = 'caregiving recipient died in past 30 days' (328 rows) is a valid codebook response - own nominal category",
        "_STATE": "keep per rule 6 (nominal, 53 FIPS codes incl. 9 = Connecticut - census miscounted FIPS 9 as sentinel); selection can prune",
        "QSTLANG": "rule 7 judgment keep: interview language (1 English / 2 Spanish) is a respondent attribute, not instrument admin (QSTVER dropped)",
        "_URBSTAT": "kept urbanicity flag (rule 7: sibling _METSTAT dropped, one per pair)",
        "_BMI5": "ENG: /100 (2 implied decimals), blank = DK/refused -> NaN; 89 rows > 80.0 kept (codebook-legal, flagged for P3/P5)",
        "_AGE80": "keep as NUM: imputed age 18-80, NO sentinels (census miscounted age 77 as sentinel); _AGEG5YR/_AGE65YR/_AGE_G dropped per rule 6",
    }
    # keep verdicts from the whitelist
    for col, spec in WHITELIST.items():
        if col in v:
            raise ValueError(f"{col} appears in both keep and drop lists")
        enc = spec["encoding"]
        verdict = "ENG" if enc == "ENG" else f"KEEP-{enc}"
        v[col] = {
            "verdict": verdict,
            "rule": "R5/R6/R7",
            "reason": "whitelisted per ADR-005 rules 5-7 (coverage >= 25%, "
                      "duplicate-pair winner or rule-7 judgment as noted in "
                      "src/preprocessing.py docstring)",
            "encoding": enc,
            "recode": str(spec["recode"]),
        }

    # keep-side reason overrides (must run AFTER the keeps exist in v)
    for col, reason in KEEP_RO.items():
        if col not in v:
            raise KeyError(f"KEEP_RO override for unknown column {col}")
        v[col]["reason"] = reason
    return v


# ---- main ----------------------------------------------------------------------
def _sanity_checks(X: pd.DataFrame, tgt: pd.DataFrame) -> dict:
    """Hard asserts + printed report. Returns a small summary dict."""
    print("\n[p2] === sanity checks ===")
    n = len(X)

    # 1. leakage blacklist can never appear
    assert not (set(BLACKLIST) & set(X.columns)), "blacklisted column in matrix"
    print(f"  ok  no blacklisted columns ({', '.join(BLACKLIST)}) in matrix")

    # 2. duplicate-pair resolutions actually hold
    assert "EXERANY2" in X.columns and "_TOTINDA" not in X.columns
    assert "HAVARTH5" in X.columns and "_DRDXAR3" not in X.columns
    assert "JOINPAI2" in X.columns and "POORHLTH" in X.columns
    print("  ok  EXERANY2 in / _TOTINDA out; HAVARTH5 in / _DRDXAR3 out; "
          "JOINPAI2 & POORHLTH present")

    # 3. shapes
    assert X.shape == (438_693, len(WHITELIST)), X.shape
    assert len(tgt) == len(X), "target frame not row-aligned"
    print(f"  ok  matrix shape = {X.shape[0]:,} x {X.shape[1]} "
          f"(expected 438,693 x {len(WHITELIST)})")

    # 4. target class rate
    n_valid = int(tgt["valid"].sum())
    n_pos = int(tgt.loc[tgt["valid"], "y"].sum())
    rate = 100.0 * n_pos / n_valid
    assert n_valid == 437_532, n_valid
    assert n_pos == 72_741, n_pos
    assert abs(rate - 16.63) < 0.01, rate
    print(f"  ok  target: {n_valid:,} valid rows, {n_pos:,} positives, "
          f"class rate {rate:.2f}% (== 16.63%)")

    # 5. residual-sentinel / value-contract scan
    violations = []
    for col, allowed in VALID_SETS.items():
        vals = X[col].dropna()
        bad = vals[~vals.isin(allowed)]
        if len(bad):
            violations.append((col, "set", bad.value_counts().head(5).to_dict()))
    for col, (lo, hi) in VALID_RANGES.items():
        vals = X[col].dropna()
        bad = vals[(vals < lo) | (vals > hi)]
        if len(bad):
            violations.append((col, "range", {"n": int(len(bad)),
                                              "min": float(bad.min()) if len(bad) else None,
                                              "max": float(bad.max()) if len(bad) else None}))
    # day-count sentinel specifics (88 must be gone, 0 = 'none')
    for col in ("PHYSHLTH", "MENTHLTH", "POORHLTH", "DRNK3GE5", "AVEDRNK3",
                "CHILDREN", "MARIJAN1"):
        assert not (X[col] == 88).any(), f"{col}: residual 88 sentinel"
        assert not X[col].isin([77, 99]).any(), f"{col}: residual 77/99"
    assert not X["ALCDAY5"].isin([777, 999, 888]).any(), "ALCDAY5 residual sentinel"
    assert not X["FRUIT2"].isin([777, 999, 555, 300]).any(), "FRUIT2 residual code"
    assert X["_STATE"].nunique() == EXPECTED_STATE_NUNIQUE
    assert X["_STATE"].isin([9]).sum() == 8_341  # FIPS 9 = Connecticut valid
    if violations:
        for col, kind, detail in violations:
            print(f"  VIOLATION {col} ({kind}): {detail}")
        raise AssertionError(f"{len(violations)} value-contract violations")
    print("  ok  value contracts hold: no residual sentinels (7/77/88/99/777/999/"
        "777777 patterns gone from day-count & ENG columns); all NOM/ORD codes "
        "within codebook sets; NUM/ENG within ranges")

    # 6. effective coverage gate (rule 5 on the keep side)
    cov = X.notna().mean()
    below = cov[cov < 0.25]
    assert below.empty, f"kept columns under 25% coverage: {below.to_dict()}"
    print(f"  ok  all {len(WHITELIST)} kept columns >= 25% effective coverage "
          f"(min = {cov.min()*100:.1f}% on {cov.idxmin()})")
    return {"n_valid": n_valid, "n_pos": n_pos, "rate": rate,
            "min_coverage_col": cov.idxmin(), "min_coverage_pct": cov.min() * 100}


def _smoke_test(X: pd.DataFrame, tgt: pd.DataFrame) -> float:
    """Prove the matrix is model-ready. TRAIN AUC on a 50k stratified subsample.

    NOT an experiment, NOT logged ('smoke - no CV').
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xv = X.loc[tgt["valid"].to_numpy()]
    yv = tgt.loc[tgt["valid"], "y"].to_numpy()
    idx = np.arange(len(Xv))
    sub_idx, _ = train_test_split(
        idx, train_size=50_000, stratify=yv, random_state=RANDOM_STATE
    )
    Xs, ys = Xv.iloc[sub_idx].reset_index(drop=True), yv[sub_idx]

    # imputation: median for NUM/ENG/ORD, most_frequent for NOM (fit on sub)
    Xi = pd.DataFrame(index=Xs.index)
    for col, spec in WHITELIST.items():
        s = Xs[col]
        if spec["encoding"] == "NOM":
            fill = s.mode().iloc[0]
            Xi[col] = s.fillna(fill).astype(np.int64).astype(str)
        else:
            Xi[col] = s.fillna(s.median())

    Xd = pd.get_dummies(Xi, columns=[c for c, sp in WHITELIST.items()
                                     if sp["encoding"] == "NOM"],
                        dtype=float)
    Xz = StandardScaler().fit_transform(Xd)
    clf = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    clf.fit(Xz, ys)
    auc = roc_auc_score(ys, clf.predict_proba(Xz)[:, 1])
    print(f"\n[p2] smoke - no CV | LogisticRegression(max_iter=500) on a "
          f"50,000-row stratified subsample ({Xd.shape[1]} columns after one-hot):")
    print(f"[p2] smoke TRAIN ROC-AUC = {auc:.4f}  (train-set AUC only; not an "
          "experiment, not logged)")
    return float(auc)


def main() -> None:
    t0 = datetime.now()
    print(f"[p2] start {t0.isoformat(timespec='seconds')}")

    df = load_raw()

    # ---- verdicts: exactly the 303 census columns, nothing else ----
    census = pd.read_csv(CENSUS_CSV)
    census_cols = list(census["col_name"])
    verdicts = build_verdicts()
    missing = sorted(set(census_cols) - set(verdicts))
    extra = sorted(set(verdicts) - set(census_cols))
    assert not missing, f"columns without verdict: {missing}"
    assert not extra, f"verdicts for unknown columns: {extra}"
    assert len(verdicts) == 303, len(verdicts)

    # cross-check rule-4 membership against census evidence
    pct = dict(zip(census["col_name"], census["pct_blank_plus_sentinel"]))
    r4_not_over90 = [c for c in SPARSE_COLS if pct[c] <= 90]
    assert not r4_not_over90, f"rule-4 cols not >90%: {r4_not_over90}"
    over90_all = {c for c, p in pct.items() if p > 90}
    over90_elsewhere = sorted(over90_all - set(SPARSE_COLS))
    print(f"[p2] census >90% blank+sentinel: {len(over90_all)} cols "
          f"({len(over90_all) - len(over90_elsewhere)} -> DROP-SPARSE, "
          f"{len(over90_elsewhere)} claimed by rules 2/3: "
          f"{', '.join(over90_elsewhere) or 'none'})")

    # ---- build ----
    X = build_clean(df)
    tgt = build_target_frame(df)
    summary = _sanity_checks(X, tgt)

    # ---- write parquets ----
    X.to_parquet(CLEAN_PARQUET, index=False)
    tgt.to_parquet(TARGET_PARQUET, index=False)
    print(f"\n[p2] wrote {CLEAN_PARQUET.name} {X.shape} and "
          f"{TARGET_PARQUET.name} {tgt.shape}")

    # ---- coverage / meta ----
    n = len(X)
    cov_eff = X.notna().mean() * 100.0
    meta = {}
    for col, spec in WHITELIST.items():
        meta[col] = {
            "encoding": spec["encoding"],
            "recode": str(spec["recode"]),
            "verdict": verdicts[col]["verdict"],
            "rule": verdicts[col]["rule"],
            "reason": verdicts[col]["reason"],
            "coverage_effective_pct": round(float(cov_eff[col]), 3),
            "pct_missing_after_recode": round(100.0 - float(cov_eff[col]), 3),
            "n_unique_after_recode": int(X[col].nunique()),
        }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[p2] wrote {META_JSON.name} ({len(meta)} columns)")

    # ---- triage CSV (census order) ----
    rows = []
    for col in census_cols:
        vd = verdicts[col]
        rows.append({
            "col_name": col,
            "verdict": vd["verdict"],
            "rule": vd["rule"],
            "reason": vd["reason"],
            "coverage_census_pct": round(100.0 - float(pct[col]), 2),
            "coverage_effective_pct": (round(float(cov_eff[col]), 2)
                                       if col in WHITELIST else ""),
            "encoding": vd["encoding"],
            "recode": vd["recode"],
        })
    triage = pd.DataFrame(rows)
    triage.to_csv(TRIAGE_CSV, index=False)
    print(f"[p2] wrote {TRIAGE_CSV.name} ({len(triage)} rows)")

    # ---- console summary ----
    by_verdict = triage["verdict"].value_counts()
    print("\n[p2] verdict counts (all 303 columns):")
    for k, v2 in by_verdict.items():
        print(f"  {k:15s} {v2}")
    enc_counts = {}
    for spec in WHITELIST.values():
        enc_counts[spec["encoding"]] = enc_counts.get(spec["encoding"], 0) + 1
    print(f"[p2] whitelist by encoding: {enc_counts} "
          f"(total {len(WHITELIST)})")
    print("[p2] top-10 missingness after recode (kept columns):")
    top10 = (100 - cov_eff).sort_values(ascending=False).head(10)
    for col, m in top10.items():
        print(f"  {col:12s} {m:6.2f}% missing  (coverage {100 - m:.2f}%)")
    print(f"[p2] overall missingness in matrix: "
          f"{X.isna().to_numpy().mean() * 100:.2f}%")
    print(f"[p2] min coverage kept column: {summary['min_coverage_col']} "
          f"{summary['min_coverage_pct']:.2f}%")

    # ---- smoke test (not logged) ----
    _smoke_test(X, tgt)

    print(f"\n[p2] done in {(datetime.now() - t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
