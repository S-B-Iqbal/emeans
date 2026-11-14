# Active context


**Phase:** 5 - Packaging - **Mode:** ACT (approved with Phase 4)
**Status:** P5.1-P5.4 done on `phase-5-packaging` (5 commits); gate report posted in issue #6.
Waiting for `approved phase 5` before merging to `main`.

## Phase 5 (done, gate pending)
- P5.1 distribution renamed to **`emeans-cluster`** (D-052; import stays `emeans`), CHANGELOG,
  README rewritten (install, quickstart, API, benchmark table incl. losses), CITATION.
- P5.2 `docs/`: conf.py + index.rst + api.rst + `auto_examples/plot_emeans_quickstart.py`;
  gate is `sphinx -W -n` (clean locally and in CI).
- P5.3 CI: `tests.yml` (3 OS x py3.10-3.13 matrix, editable, coverage >=90%, nightly sklearn),
  `docs.yml`, `wheels.yml` (cibuildwheel artefacts only). 14/14 test jobs green.
- P5.4 `docs/release-runbook.md`: bump, build, twine check, tag, upload, yank — printed, never run.
- Fixes found by CI: gallery sources were gitignored; MSVC OpenMP linked a missing vcomp DLL
  (D-056 - Windows builds serial; the meson override had to sit above `subdir('emeans')`).

## Phases 1-4 (done, merged)
- P1 pure NumPy reference: 107 tests, 109 doctests, 99% coverage.
- P2 compliance: `parametrize_with_checks` green (4 justified xfails, D-041).
- P3 Cython/OpenMP kernels + `_backend` switch (D-044/D-045/D-046).
- P4 benchmarks: `benchmarks/report.md` + CSV + 3 plots; e-means hits Lloyd@25 in 2-4 epochs,
  loses to a tuned MiniBatchKMeans on 3/4 laptop configs.
- `main` = `908f82b` (Phase 4 merge).

## Phase 6 (next, drafts only, never posted)
contrib inclusion request + comment for scikit-learn#35383 + Discussion + MiniBatchKMeans PR
sketch, saved under `docs/upstream/`.

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"`.
- Repo-local git identity `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031);
  pushing `origin` branches is pre-authorised (D-030). Tags, `gh release`, PyPI upload,
  visibility changes and any write to another repository need explicit confirmation each time.
- Generated docs output (`docs/generated/`, `docs/sg_execution_times.rst`) is gitignored;
  `docs/auto_examples/` holds the sources and is tracked.
