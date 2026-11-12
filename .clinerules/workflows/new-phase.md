# /new-phase.md — open a phase (only after approval of the previous one)

1. Read every memory-bank file and `tasks/TASKS.md` (rule 02).
2. Confirm the previous phase is approved in `memory-bank/progress.md`; if not, STOP and ask.
3. In PLAN mode, write into `memory-bank/activeContext.md`:
   files to create/change · functions · tests (acceptance criteria) · risks · open questions touched.
4. Ask the user to approve the plan.
5. On approval: switch to ACT, `git checkout -b phase-N-<slug>`.
6. Work `tasks/TASKS.md` items top-down; after each green item: commit (rule 08), tick the box,
   one-line note in progress.md.
