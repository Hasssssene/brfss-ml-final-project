# ADR-003 — Agent team and git protocol

- **date:** 2026-10-25
- **status:** accepted
- **tags:** #adr #process
- **scope:** working process, commits, vault

## Context
Multi-agent development (AI workers) risks undocumented work, inconsistent history, and unreviewed results entering the repo. The course also requires an honest AI-assistance disclosure.

## Decision
Five specialist agents (defined in `.opencode/agent/`): `ml-engineer` (code+experiments), `data-analyst` (EDA/interpretation/stats), `report-writer` (report only), `vault-scribe` (vault + **sole committer**), `qa-reviewer` (read-only auditor). The main session acts as coordinator and approves ADRs.

**Milestone protocol:** worker → (optional) QA audit → vault-scribe documents → commits (`exp:`/`feat:`/`report:`/`docs:`) → pushes. The scribe must refuse to commit undocumented work. Emergency exception: coordinator may make a single `wip:` commit, superseded at the next milestone.

## Alternatives considered
- Any worker may commit — faster, but breaks "no documentation, no commit" and yields noisy history. Rejected.
- Human-only commits — student time is the scarcest resource before the deadline. Rejected.

## Consequences
- Single choke point enforces vault completeness and uniform commit style.
- First bootstrap commits (2026-10-25) are made by the coordinator before the scribe agent is loaded in a session — noted here for provenance.
- `qa-reviewer` audits are mandatory before submission (see [[Rubric Checklist]]).
