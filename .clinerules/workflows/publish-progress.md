# /publish-progress.md — keep GitHub in step with local progress (D-030)

Pre-authorised repository: **`S-B-Iqbal/emeans`** (private until the Phase 1 gate, then public —
D-032). Never run a `gh` write command against any other repository.

## After every green task (autonomous)
1. The task's acceptance test passes.
2. Memory bank: one line in `memory-bank/progress.md`; tick the box in `tasks/TASKS.md`.
3. `pre-commit run --all-files`, `pytest --pyargs emeans -q`, `pytest --doctest-modules emeans -q`.
4. `git add -A && git commit -m "<conventional message>"` (rule 08).
5. `git push origin <current phase branch>`.
6. Check CI: `gh run list --repo S-B-Iqbal/emeans --limit 3`. On failure, fix forward with a
   **new** commit and push again — never force-push.
7. Tick the matching item on the phase issue and post the test summary line:
   `gh issue comment <n> --repo S-B-Iqbal/emeans --body "P1.2 done — pytest: 5 passed"`.

## At a phase gate (stop before merging)
1. Run the gate commands of `/phase-gate.md`; push the branch; comment the gate summary on the
   phase issue.
2. Ask exactly: **“Approve Phase N and start Phase N+1?”** and stop.
3. Only after `approved phase N`:
   - `git checkout main && git merge --no-ff phase-N-<slug> && git push origin main`
   - close the phase issue (`gh issue close <n>`), open the next phase issue with its checklist
   - update the README roadmap status table and push.

## Never (rule 01)
`git push --force`, history rewrites after the first push, `git tag`, `gh release`,
`twine upload`, visibility/collaborator changes, `gh secret`/`gh variable`, deleting the
repository or branches, and any write action on another repository.
