# /phase-gate.md — close a phase

1. Run and capture the summary line of each:
   - `pre-commit run --all-files`
   - `pytest --pyargs emeans -q`
   - `pytest --doctest-modules emeans -q`
   - Phase ≥ 2: `pytest emeans/tests/test_common.py -q`
   - Phase ≥ 3: `pytest emeans/tests/test_backends.py -q`
2. Update `memory-bank/progress.md` (what works / what's left / known issues) and
   `memory-bank/activeContext.md` (proposed plan for next phase).
3. Tick completed items in `tasks/TASKS.md`.
4. Post a summary to the user:
   - Done · Tests (paste summary lines) · Deviations from the paper · New open questions (Q-ids)
   - Risks · Proposed next-phase plan
5. Ask exactly: **"Approve Phase N and start Phase N+1?"**
6. STOP. Do nothing else until explicit approval.
