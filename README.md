# BRFSS 2021 — Predicting Fair/Poor General Health (ML Final Project)

Binary classification on the CDC Behavioral Risk Factor Surveillance System 2021 dataset
(~438k respondents, 300+ features): predict whether a respondent reports **Fair or Poor**
general health (`GENHLTH ∈ {4,5}`) vs **Good or better** (`{1,2,3}`), evaluated with
**ROC-AUC under stratified 10-fold cross-validation** (fixed folds, `random_state=42`).

## Setup

```bash
python -m pip install -r requirements.txt
```

Raw data (`data/raw/LLCP2021.csv`) is **not** in the repo. Download from the
[UA server](https://www.dlsi.ua.es/~juanra/UA/datasets/LLCP2021.csv.tar.gz) and extract
into `data/raw/`. Variable documentation: BRFSS 2021 codebook (`docs/codebook.pdf`,
not committed — see [official site](https://www.cdc.gov/brfss/annual_data/annual_2021.html)).

## Usage

```bash
python src/profile_data.py      # data profiling + column triage inputs
python deliverables/code.py     # full pipeline end-to-end (final deliverable)
```

## Structure

| Path | Contents |
|---|---|
| `src/` | Pipeline modules (config, preprocessing, experiments, tuning, analysis) |
| `results/` | `experiments_log.csv` (append-only), per-run artifacts, analysis notes |
| `figures/` | All report figures |
| `deliverables/` | `code.py` + `report/report.pdf` |
| `vault/` | Obsidian vault — decisions (ADRs), experiment notes, daily logs |
| `docs/` | Course guidelines, codebook reference |

`AGENTS.md` documents the working rules (reproducibility, leakage control, commit protocol).

## Reproducibility

- Every experiment: `StratifiedKFold(10, shuffle=True, random_state=42)` from `src/config.py`.
- Every run logged in `results/experiments_log.csv` with parameters and 10 fold scores.
- Course deliverables per `docs/guidelines.md`: technical report, `code.py`, screencast.
