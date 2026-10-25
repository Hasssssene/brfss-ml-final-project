# Rules & Conventions (vault mirror)

> [!important] The binding version of all project rules is `AGENTS.md` at the repo root. This note only adds vault-specific conventions.

## Vault protocol
- The **vault-scribe** is the only agent that writes here, and only as part of a milestone (document → commit → push).
- One note per experiment, named `EXP-xxx <short name>` in [[20 Experiments]].
- One ADR per methodology decision, numbered sequentially in [[30 Decisions]]. Format: Context → Decision → Alternatives considered → Consequences.
- Daily logs are append-only; never rewrite an old day.

## Note hygiene
- Tags: `#exp` `#adr` `#daily` `#data`
- Numbers in notes must match `results/experiments_log.csv` — if it is not logged, it does not exist.
- Figures referenced from `figures/` (repo) or `90 Attachments/` (copies embedded in the vault).
