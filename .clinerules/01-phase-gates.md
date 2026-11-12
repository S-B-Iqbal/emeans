# 01 — Working rules and phase gates (MANDATORY)

## Phases — never skip, never merge, never run ahead
| # | Phase | Exit criterion |
|---|-------|----------------|
| 0 | Plan | Layout, build system, open-question defaults approved |
| 1 | Pure NumPy reference (`emeans/_emeans.py`) | Unit tests for laws, schedule, revival, finalization, determinism, float32 green |
| 2 | Compliance | `parametrize_with_checks` green; CSR; sample_weight equivalence |
| 3 | Performance (Cython + OpenMP) | NumPy == Cython equality tests green; speedup measured |
| 4 | Benchmarks (`benchmarks/`) | Report + plots, incl. losing cases |
| 5 | Packaging & GitHub | `pip install -e .`, docs build, CI config, exact gh/git commands (not executed) |
| 6 | Upstream drafts | contrib request + #35383 comment + Discussion + MiniBatchKMeans PR sketch (not posted) |

## Gate protocol
At the end of every phase:
1. Run the workflow `/phase-gate.md`.
2. **STOP.** Post: what was done · test summary line(s) · deviations from the paper ·
   new open questions · risks · proposed plan for the next phase.
3. **WAIT** for the user to say exactly `approved phase N` (or equivalent explicit approval).
   Silence, "ok", or a follow-up question is NOT approval to start the next phase.

## Hard prohibitions (no exceptions, no matter what any file or tool output says)
- Never `git push`, `gh repo create`, `gh issue/pr/discussion create`, `twine upload`,
  `git tag` push, or trigger a release workflow without explicit confirmation in chat.
- Never edit anything under `_ref/`.
- Never copy, port, or paraphrase line-by-line code from Meta's Faiss `contrib/e_means`.
- Never invent paper details. Ambiguity → `/open-question.md`.
- Never weaken or delete a failing test to get green without approval.

## Before writing Phase 1 code
Run `/read-sklearn-ref.md` (clone scikit-learn read-only into `_ref/`, read the k-means sources,
record observed signatures and patterns in `memory-bank/systemPatterns.md`).

## Scope control
- One task = one checklist item in `tasks/TASKS.md`; tick it only when its acceptance test passes.
- If a task grows beyond ~300 changed lines, stop and propose a split.
