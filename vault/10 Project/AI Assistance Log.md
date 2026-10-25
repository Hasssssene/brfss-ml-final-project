# AI Assistance Log

> [!important] Running log for the mandatory "Ethics and Use of Generative AI" report section. Every substantial AI contribution gets an entry: tool, purpose, how output was used. The report-writer drafts the final statement FROM THIS LOG.

| # | Date | Tool | Purpose | How the output was used |
|---|------|------|---------|------------------------|
| 1 | 2026-10-25 | OpenCode agent (GLM, Z.ai) | Project planning: read guidelines + codebook, framed binary target, designed pipeline, agent team, repo and vault structure | Plan reviewed and approved by the student; structure adopted for the project |
| 2 | 2026-10-25 | OpenCode agent (GLM, Z.ai) | Repository scaffolding: .gitignore, README, AGENTS.md rules, vault seed content, src/config.py | Files reviewed and committed as the initial project setup |
| 3 | 2026-10-25 | OpenCode agent (GLM, Z.ai) | Interpreting the BRFSS codebook structure (missing-value codes, section layout, GENHLTH distribution) | Used to draft the preprocessing rules and leakage watchlist; to be verified against the data in EXP-000 |
| 4 | 2026-10-25 | OpenCode agent (GLM, Z.ai) | EXP-000 profiling script + analysis of missingness/association/leakage probe | Script committed as `src/profile_data.py`; findings documented in vault and to be verified in P2 matrix build |
| 5 | 2026-10-25 | OpenCode agent (GLM, Z.ai) | P2 whitelist construction + preprocessing implementation (src/preprocessing.py) + sentinel/codebook cross-verification | Code reviewed via sanity gates; all 303 column verdicts in results/profiling/triage_decisions.csv; decisions ratified as ADR-005 |
