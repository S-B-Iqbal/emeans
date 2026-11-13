# Active context


**Phase:** 2 - Compliance - **Mode:** ACT (Phase 2 plan approved with Phase 1)
**Status:** Phase 1 approved, merged to `main` and the repo is public; P2.1 in
progress on `phase-2-compliance` (issue #3).

## Phase 1 (done)
107 unit tests, 109 doctests, 99% coverage, CI green (wheel + editable jobs).
Modules: `_scaling_laws` (Eq. 6-7), `_schedule` (Eq. 5), `_numpy_backend`
(Eq. 1/3, dense + CSR), `_revival` (App. H.1), `_compat`, `_validation`,
`_emeans` (`EMeans`: fit, finalization, public API, partial_fit).

## Phase 2 plan (approved)
1. `parametrize_with_checks([EMeans(...)])` green, with any justified
   `expected_failed_checks` recorded in the decision log (P2.1).
2. CSR coverage: `test_sparse.py` with dense == sparse (same seed, explicit
   params) for centres, labels and inertia (P2.2).
3. `test_sample_weight.py`: exact equivalence in the `partial_fit(alpha0=1.0)`
   regime, weight-scale invariance, and a statistical duplicated-rows test
   (P2.3).
4. Validation messages matched to sklearn style; error/warning tests (P2.4).

## Phase 0 (done, approved)
meson-python skeleton, `emeans/__init__.py` + `py.typed` + smoke test, LICENSE/README/CITATION.cff,
minimal CI, decisions D-001…D-036, sklearn reference recorded in `systemPatterns.md`
(@7b2b14e0a89eedfa514dc9885e35a0ecade20cd0); repo private, issue #1 closed at the gate.

## Publishing (D-030…D-034)
- `origin` = `S-B-Iqbal/emeans` (private until the Phase 1 gate, then public — D-032); loop defined
  in `.clinerules/workflows/publish-progress.md`: push after every green task, tick the phase issue,
  merge to `main` only after a gate is approved.
- CI: `.github/workflows/tests.yml` (ubuntu, py3.10) on every push (D-034); full matrix in Phase 5.
  Green on `fix(ci): keep editable installs non-isolated…` (the very first run exposed the
  meson-python editable/build-isolation trap).
- Phase 0.6 (repo creation, LICENSE/README/CITATION, CI) is tracked as P0.6–P0.8 in `TASKS.md` and
  in issue #1.

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"` where
  the resolved batch size would be large.
- Repo-local git identity is `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031) and
  pushing to `origin` is pre-authorised (D-030); tags, releases, PyPI and other repositories are not.
