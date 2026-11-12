# 02 — Memory Bank protocol

My memory resets completely between sessions. The Memory Bank is my only link to prior work.

## At the start of EVERY task (not optional)
Read ALL of these, in order, before anything else:
1. `memory-bank/projectbrief.md`
2. `memory-bank/productContext.md`
3. `memory-bank/systemPatterns.md`
4. `memory-bank/techContext.md`
5. `memory-bank/activeContext.md`
6. `memory-bank/progress.md`
7. `memory-bank/openQuestions.md`
8. `memory-bank/decisionLog.md`
9. `tasks/TASKS.md`

Then state in one line: `Phase N · task <id> · mode PLAN|ACT`.

## When to update
| File | Update when |
|------|-------------|
| activeContext.md | End of every working session; whenever focus changes |
| progress.md | A checklist item, test suite, or phase completes; a known issue appears/disappears |
| systemPatterns.md | An architectural pattern is adopted or changed |
| techContext.md | Tooling, versions, commands, CI matrix change |
| openQuestions.md | New ambiguity found (Q-NNN) |
| decisionLog.md | User approves a resolution (D-NNN) — move the Q there, keep the Q id |
| projectbrief.md, productContext.md | Only at milestones, with user approval |

## "update memory bank"
When the user says **update memory bank** (or runs `/update-memory-bank.md`), review EVERY file,
even ones that need no change. Prioritise activeContext.md and progress.md.

## Hygiene
- activeContext.md stays short (< 60 lines). Move history into progress.md.
- Record facts, decisions, commands, and test results — not narrative.
- Never store secrets, tokens, PyPI credentials, or dataset auth.
