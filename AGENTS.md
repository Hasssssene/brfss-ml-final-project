# AGENTS.md — Project Rules (binding for all agents and contributors)

## Mission
Binary classification on BRFSS 2021: predict **y = 1 if GENHLTH ∈ {4,5}** (Fair/Poor) vs **y = 0 if GENHLTH ∈ {1,2,3}**.
Metric: **ROC-AUC**. Validation: **stratified 10-fold CV on identical folds for every model**.
Deliverables: `deliverables/code.py` (single file, runs end-to-end), `deliverables/report/report.pdf`, screencast.
**Deadline: November 8, 2026, 23:59 (C2 call).**

## Repository layout
- `data/raw/` — immutable raw data (gitignored). `data/interim/` — derived checkpoints (gitignored).
- `src/` — pipeline modules. `src/analysis/` — throwaway analysis snippets.
- `results/` — `experiments_log.csv` (append-only) + per-run artifacts + `analysis/` notes. Committed.
- `figures/` — all figures (PNG, dpi≥150). Committed.
- `deliverables/` — final `code.py` and `report/`. Committed.
- `vault/` — Obsidian vault (project memory). Committed.
- `docs/` — guidelines, codebook (codebook gitignored).

## Agent team & authority
| Agent | Writes to | Git |
|---|---|---|
| coordinator (main session) | architecture, ADR approval, dispatch | emergency `wip:` commits only |
| `ml-engineer` | `src/`, `results/`, `figures/`, `data/interim/` | **never** |
| `data-analyst` | `results/analysis/`, `figures/`, `src/analysis/` | **never** |
| `report-writer` | `deliverables/report/` | **never** |
| `vault-scribe` | `vault/`, `.gitignore`, `README` prose | **exclusively** |
| `qa-reviewer` | nothing (read-only) | never |

**Milestone protocol**: worker finishes → (optional per milestone, mandatory pre-submission) `qa-reviewer` audits → `vault-scribe` documents then commits (`exp:` / `feat:` / `report:` / `docs:`) and pushes. A commit without its vault note must not exist.

## Reproducibility rules
1. `RANDOM_STATE = 42` everywhere. No other seed values allowed in committed code.
2. The only CV splitter: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` via `src/config.py:make_folds()`. Every model, every selection, every SMOTE pipeline — same folds.
3. `results/experiments_log.csv` is append-only: one row per run (exp_id, timestamp, model, params, feature_set, imbalance_strategy, 10 fold AUCs, mean AUC, notes).
4. Raw data is immutable. Cleaning checkpoints go to `data/interim/` (parquet) so CV never re-cleans.
5. Any randomness in feature selection / resampling also uses `RANDOM_STATE`.

## Data integrity rules
1. **Leakage blacklist** (must never appear as a feature): see `vault/40 Data Notes/Leakage watchlist.md` — currently `_RFHLTH` (it IS the target), `_PHYS14D`, `_MENT14D`, plus duplicate-calculated pairs listed there.
2. Missing sentinels per BRFSS codebook: 7/77/777 = Don't know, 9/99/999 = Refused, blank = not asked/missing, 88 = "none" for day-count variables, 7777/9999 for height/weight. Recode before modeling; never treat sentinels as numeric values.
3. Rows with target ∈ {7, 9, blank} are dropped (~1,161 rows), not imputed.
4. The dataset (any version) never enters git history.

## Methodology change control
Target, folds, metrics, feature sets, and preprocessing decisions change only via an **ADR** in `vault/30 Decisions/` approved by the coordinator. Workers implement; they do not decide.

## Report rules
- Every number cites an EXP id traceable to `results/experiments_log.csv`.
- Report structure mirrors the rubric (`vault/10 Project/Rubric Checklist.md`).
- Final section "Ethics and Use of Generative AI" is drafted from `vault/10 Project/AI Assistance Log.md` — honest disclosure is mandatory per course policy; concealment risks a zero.

## AI assistance logging
Every substantial AI contribution (code generation, analysis interpretation, report prose) gets a one-line entry in `vault/10 Project/AI Assistance Log.md`. Cheap now, disqualifying later if missing.

## Git conventions
- Conventional commits: `feat:`, `exp:`, `report:`, `docs:`, `chore:`, emergency `wip:`.
- Push after every milestone commit. Never force-push. Never rewrite history.
