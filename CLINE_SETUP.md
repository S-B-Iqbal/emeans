# Using this Cline scaffold

## Install
1. Unzip into the root of your new repo (e.g. `~/code/emeans/`).
   Hidden files matter: `.clinerules/`, `.clineignore`, `.gitignore`, `.pre-commit-config.yaml`.
2. `bash scripts/bootstrap.sh` → venv, dev deps, read-only `_ref/scikit-learn`, paper PDF, git, hooks.
3. Open the folder in VS Code with Cline. Confirm in Cline's Rules panel that all `.clinerules/*.md`
   are toggled on (06, 07, 09 are path-conditional and activate only for matching files).

## First prompt (PLAN mode)
> Read the memory bank, `docs/PAPER_NOTES.md`, and `memory-bank/openQuestions.md`.
> Present the Phase 0 plan and your recommendation on each open question. Do not write code.

Then approve/amend the open questions, e.g. "approve Q-001..Q-011 as proposed; Q-012: use X".
Cline records them in `decisionLog.md`.

## Workflows (type in the Cline chat)
| Command | When |
|---|---|
| `/read-sklearn-ref.md` | Once, before Phase 1 code |
| `/new-phase.md` | Start a phase after approval |
| `/open-question.md` | Any paper ambiguity |
| `/update-memory-bank.md` | End of a session, or after a big change |
| `/phase-gate.md` | Close a phase; Cline stops and waits |
| `/release.md` | Phase 5+, prints commands, never runs push/publish itself |

## Approval phrases Cline is told to wait for
`approved phase N` — anything less is not a go-ahead.

## Layout
```
.clinerules/            00–10 rules + workflows/
memory-bank/            projectbrief, productContext, systemPatterns, techContext,
                        activeContext, progress, openQuestions, decisionLog
tasks/TASKS.md          per-phase checklist with IDs (P1.4 etc.)
docs/PAPER_NOTES.md     paper digest (Eq./Table/App. references)
docs/REFERENCES.md      what each reference contributes
scripts/bootstrap.sh    local setup
```
